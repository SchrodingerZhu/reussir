//===----------------------------------------------------------------------===//
//
// Part of the Reussir Project, dual licensed under the Apache License v2.0 or
// the MIT License.
// See https://github.com/reussir-lang/reussir/blob/main/LICENSE for license
// information.
// SPDX-License-Identifier: Apache-2.0 OR MIT
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the JSON streamer used by rrc for token-reuse remarks.
///
//===----------------------------------------------------------------------===//

#include "Reussir-c/Passes.h"

#include <algorithm>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <system_error>
#include <tuple>
#include <utility>
#include <vector>

#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/JSON.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/CAPI/IR.h>
#include <mlir/CAPI/Support.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/Remarks.h>

namespace {

using LocationId = uint32_t;

enum class LocationKind : uint8_t {
  File,
  Name,
  CallSite,
  Fused,
  Unknown,
  Opaque,
};

enum class DecisionKind : uint8_t { Reuse, Allocate };
enum class ReuseStrategy : uint8_t { None, Ensure, Realloc };
enum class AllocationReason : uint8_t {
  None,
  NoAvailableToken,
  NoCompatibleToken,
};

struct LocationNode {
  LocationKind kind = LocationKind::Unknown;
  // File path, NameLoc name, or the printed form of an opaque location. These
  // are the only location payloads that cannot be represented by numeric IDs.
  std::string text;
  uint64_t startLine = 0;
  uint64_t startColumn = 0;
  uint64_t endLine = 0;
  uint64_t endColumn = 0;
  llvm::SmallVector<LocationId, 2> children;
};

struct DecisionKey {
  DecisionKind kind = DecisionKind::Allocate;
  LocationId sink = 0;
  std::optional<LocationId> source;
  ReuseStrategy strategy = ReuseStrategy::None;
  AllocationReason reason = AllocationReason::None;
  int64_t score = 0;
  uint64_t availableTokens = 0;
  uint64_t compatibleTokens = 0;

  bool operator<(const DecisionKey &other) const {
    return std::tie(kind, sink, source, strategy, reason, score,
                    availableTokens, compatibleTokens) <
           std::tie(other.kind, other.sink, other.source, other.strategy,
                    other.reason, other.score, other.availableTokens,
                    other.compatibleTokens);
  }
};

struct Decision {
  std::set<std::string> instances;
  uint64_t occurrences = 0;
};

struct InternedLocation {
  LocationId id;
  // References a std::map key, whose address remains stable across insertion.
  llvm::StringRef canonicalKey;
};

void appendKeyPart(std::string &key, llvm::StringRef part) {
  key += std::to_string(part.size());
  key.push_back(':');
  key.append(part.data(), part.size());
}

std::optional<int64_t> parseSignedMetric(llvm::StringRef value) {
  int64_t parsed = 0;
  if (value.getAsInteger(10, parsed))
    return std::nullopt;
  return parsed;
}

std::optional<uint64_t> parseUnsignedMetric(llvm::StringRef value) {
  uint64_t parsed = 0;
  if (value.getAsInteger(10, parsed))
    return std::nullopt;
  return parsed;
}

llvm::StringRef decisionKindName(DecisionKind kind) {
  switch (kind) {
  case DecisionKind::Reuse:
    return "reuse";
  case DecisionKind::Allocate:
    return "allocate";
  }
  return "";
}

llvm::StringRef strategyName(ReuseStrategy strategy) {
  switch (strategy) {
  case ReuseStrategy::Ensure:
    return "ensure";
  case ReuseStrategy::Realloc:
    return "realloc";
  case ReuseStrategy::None:
    return "";
  }
  return "";
}

llvm::StringRef reasonName(AllocationReason reason) {
  switch (reason) {
  case AllocationReason::NoAvailableToken:
    return "no-available-token";
  case AllocationReason::NoCompatibleToken:
    return "no-compatible-token";
  case AllocationReason::None:
    return "";
  }
  return "";
}

class TokenReuseJsonStreamer final
    : public mlir::remark::detail::MLIRRemarkStreamerBase {
public:
  explicit TokenReuseJsonStreamer(std::unique_ptr<llvm::raw_fd_ostream> output)
      : output(std::move(output)) {}

  void streamOptimizationRemark(
      const mlir::remark::detail::Remark &remark) override {
    if (remark.getCategoryName() != "TokenReuse")
      return;

    std::lock_guard lock(mutex);
    DecisionKey incoming;
    if (remark.getRemarkName() == "TokenReused")
      incoming.kind = DecisionKind::Reuse;
    else if (remark.getRemarkName() == "TokenNotReused")
      incoming.kind = DecisionKind::Allocate;
    else
      return;

    incoming.sink = internLocation(remark.getLocation()).id;
    for (const auto &arg : remark.getArgs()) {
      if (arg.key == "Source" && arg.hasAttribute()) {
        if (auto loc = llvm::dyn_cast<mlir::LocationAttr>(arg.getAttribute()))
          incoming.source = internLocation(mlir::Location(loc)).id;
      } else if (arg.key == "Strategy") {
        if (arg.val == "ensure")
          incoming.strategy = ReuseStrategy::Ensure;
        else if (arg.val == "realloc")
          incoming.strategy = ReuseStrategy::Realloc;
      } else if (arg.key == "Reason") {
        if (arg.val == "no-available-token")
          incoming.reason = AllocationReason::NoAvailableToken;
        else if (arg.val == "no-compatible-token")
          incoming.reason = AllocationReason::NoCompatibleToken;
      } else if (arg.key == "Score") {
        if (auto value = parseSignedMetric(arg.val))
          incoming.score = *value;
      } else if (arg.key == "AvailableTokens") {
        if (auto value = parseUnsignedMetric(arg.val))
          incoming.availableTokens = *value;
      } else if (arg.key == "CompatibleTokens") {
        if (auto value = parseUnsignedMetric(arg.val))
          incoming.compatibleTokens = *value;
      }
    }

    Decision &decision = decisions.try_emplace(incoming).first->second;
    ++decision.occurrences;
    decision.instances.emplace(remark.getFunction());
  }

  void finalize() override {
    std::lock_guard lock(mutex);
    if (finalized)
      return;
    finalized = true;
    writeReport();
  }

private:
  InternedLocation internLocation(mlir::Location loc) {
    LocationNode node;
    std::string key;

    if (auto file = llvm::dyn_cast<mlir::FileLineColRange>(loc)) {
      node.kind = LocationKind::File;
      node.text = file.getFilename().getValue().str();
      node.startLine = file.getStartLine();
      node.startColumn = file.getStartColumn();
      node.endLine = file.getEndLine();
      node.endColumn = file.getEndColumn();
      key.push_back(static_cast<char>(LocationKind::File));
      appendKeyPart(key, node.text);
      for (uint64_t coordinate :
           {node.startLine, node.startColumn, node.endLine, node.endColumn}) {
        key.push_back('#');
        key += std::to_string(coordinate);
      }
    } else if (auto name = llvm::dyn_cast<mlir::NameLoc>(loc)) {
      node.kind = LocationKind::Name;
      node.text = name.getName().getValue().str();
      InternedLocation child = internLocation(name.getChildLoc());
      node.children.push_back(child.id);
      key.push_back(static_cast<char>(LocationKind::Name));
      appendKeyPart(key, node.text);
      appendKeyPart(key, child.canonicalKey);
    } else if (auto callSite = llvm::dyn_cast<mlir::CallSiteLoc>(loc)) {
      node.kind = LocationKind::CallSite;
      InternedLocation callee = internLocation(callSite.getCallee());
      InternedLocation caller = internLocation(callSite.getCaller());
      node.children.push_back(callee.id);
      node.children.push_back(caller.id);
      key.push_back(static_cast<char>(LocationKind::CallSite));
      appendKeyPart(key, callee.canonicalKey);
      appendKeyPart(key, caller.canonicalKey);
    } else if (auto fused = llvm::dyn_cast<mlir::FusedLoc>(loc)) {
      llvm::SmallVector<InternedLocation, 2> children;
      for (mlir::Location child : fused.getLocations()) {
        children.push_back(internLocation(child));
        node.children.push_back(children.back().id);
      }
      // Debug metadata commonly wraps one source location in a FusedLoc.
      // Treat that wrapper as the source location itself so `-g` does not
      // defeat deduplication or split generic instances into separate sites.
      if (children.size() == 1)
        return children.front();
      node.kind = LocationKind::Fused;
      key.push_back(static_cast<char>(LocationKind::Fused));
      for (const InternedLocation &child : children)
        appendKeyPart(key, child.canonicalKey);
    } else if (llvm::isa<mlir::UnknownLoc>(loc)) {
      node.kind = LocationKind::Unknown;
      key.push_back(static_cast<char>(LocationKind::Unknown));
    } else {
      node.kind = LocationKind::Opaque;
      llvm::raw_string_ostream text(node.text);
      loc.print(text);
      text.flush();
      key.push_back(static_cast<char>(LocationKind::Opaque));
      appendKeyPart(key, node.text);
    }

    auto [it, inserted] = locationIds.try_emplace(std::move(key), 0);
    if (inserted) {
      it->second = static_cast<LocationId>(locations.size());
      locations.push_back(std::move(node));
    }
    return {it->second, llvm::StringRef(it->first)};
  }

  void writeReport() {
    std::set<std::string> files;
    for (const LocationNode &location : locations)
      if (location.kind == LocationKind::File)
        files.insert(location.text);

    std::map<std::string, uint64_t> fileIds;
    uint64_t nextId = 0;
    for (const std::string &file : files)
      fileIds.emplace(file, nextId++);

    // Internal IDs are optimized for insertion. Remap them using the sorted
    // canonical keys so parallel pass scheduling cannot perturb JSON output.
    std::vector<uint64_t> outputLocationIds(locations.size());
    nextId = 0;
    for (const auto &[_, internalId] : locationIds)
      outputLocationIds[internalId] = nextId++;

    struct OrderedDecision {
      const DecisionKey *key;
      const Decision *decision;
    };
    std::vector<OrderedDecision> orderedDecisions;
    orderedDecisions.reserve(decisions.size());
    for (const auto &[key, decision] : decisions)
      orderedDecisions.push_back({&key, &decision});
    auto outputSourceId = [&](const DecisionKey &key) {
      return key.source
                 ? std::optional<uint64_t>(outputLocationIds[*key.source])
                 : std::nullopt;
    };
    std::sort(
        orderedDecisions.begin(), orderedDecisions.end(),
        [&](const OrderedDecision &left, const OrderedDecision &right) {
          const DecisionKey &lhs = *left.key;
          const DecisionKey &rhs = *right.key;
          return std::make_tuple(lhs.kind, outputLocationIds[lhs.sink],
                                 outputSourceId(lhs), lhs.strategy, lhs.reason,
                                 lhs.score, lhs.availableTokens,
                                 lhs.compatibleTokens) <
                 std::make_tuple(rhs.kind, outputLocationIds[rhs.sink],
                                 outputSourceId(rhs), rhs.strategy, rhs.reason,
                                 rhs.score, rhs.availableTokens,
                                 rhs.compatibleTokens);
        });

    llvm::json::OStream json(*output, 2);
    json.object([&] {
      json.attribute("schema", "reussir.token-reuse.v1");
      json.attributeObject("coordinates", [&] {
        json.attribute("line_base", int64_t{1});
        json.attribute("column_base", int64_t{1});
        json.attribute("column_unit", "utf-8-byte");
        json.attribute("end", "exclusive");
      });
      json.attributeArray("files", [&] {
        for (const std::string &file : files)
          json.value(file);
      });
      json.attributeArray("locations", [&] {
        for (const auto &[_, internalId] : locationIds) {
          const LocationNode &location = locations[internalId];
          json.object([&] {
            switch (location.kind) {
            case LocationKind::File:
              json.attribute("kind", "file");
              json.attribute("file",
                             static_cast<int64_t>(fileIds.at(location.text)));
              json.attributeObject("start", [&] {
                json.attribute("line",
                               static_cast<int64_t>(location.startLine));
                json.attribute("column",
                               static_cast<int64_t>(location.startColumn));
              });
              json.attributeObject("end", [&] {
                json.attribute("line", static_cast<int64_t>(location.endLine));
                json.attribute("column",
                               static_cast<int64_t>(location.endColumn));
              });
              break;
            case LocationKind::Name:
              json.attribute("kind", "name");
              json.attribute("name", location.text);
              json.attribute("child",
                             static_cast<int64_t>(
                                 outputLocationIds[location.children[0]]));
              break;
            case LocationKind::CallSite:
              json.attribute("kind", "callsite");
              json.attribute("callee",
                             static_cast<int64_t>(
                                 outputLocationIds[location.children[0]]));
              json.attribute("caller",
                             static_cast<int64_t>(
                                 outputLocationIds[location.children[1]]));
              break;
            case LocationKind::Fused:
              json.attribute("kind", "fused");
              json.attributeArray("locations", [&] {
                for (LocationId child : location.children)
                  json.value(static_cast<int64_t>(outputLocationIds[child]));
              });
              break;
            case LocationKind::Unknown:
              json.attribute("kind", "unknown");
              break;
            case LocationKind::Opaque:
              json.attribute("kind", "opaque");
              json.attribute("text", location.text);
              break;
            }
          });
        }
      });
      json.attributeArray("decisions", [&] {
        for (const OrderedDecision &ordered : orderedDecisions) {
          const DecisionKey &key = *ordered.key;
          const Decision &decision = *ordered.decision;
          json.object([&] {
            json.attribute("kind", decisionKindName(key.kind));
            json.attribute("sink",
                           static_cast<int64_t>(outputLocationIds[key.sink]));
            if (key.source)
              json.attribute("source", static_cast<int64_t>(
                                           outputLocationIds[*key.source]));
            if (key.strategy != ReuseStrategy::None)
              json.attribute("strategy", strategyName(key.strategy));
            if (key.reason != AllocationReason::None)
              json.attribute("reason", reasonName(key.reason));
            if (key.kind == DecisionKind::Reuse)
              json.attribute("score", key.score);
            json.attribute("available_tokens",
                           static_cast<int64_t>(key.availableTokens));
            json.attribute("compatible_tokens",
                           static_cast<int64_t>(key.compatibleTokens));
            json.attributeArray("instances", [&] {
              for (const std::string &instance : decision.instances)
                json.value(instance);
            });
            json.attribute("occurrences",
                           static_cast<int64_t>(decision.occurrences));
          });
        }
      });
    });
    *output << '\n';
    output->flush();
  }

  std::unique_ptr<llvm::raw_fd_ostream> output;
  std::mutex mutex;
  // The canonical string is required for structural location deduplication and
  // deterministic ordering. Decisions themselves use compact numeric IDs.
  std::map<std::string, LocationId> locationIds;
  std::vector<LocationNode> locations;
  std::map<DecisionKey, Decision> decisions;
  bool finalized = false;
};

} // namespace

bool reussirContextEnableTokenReuseRemarks(MlirContext context,
                                           MlirStringRef outputPath) {
  mlir::MLIRContext *ctx = unwrap(context);
  if (ctx->getRemarkEngine())
    return false;

  std::error_code error;
  auto output = std::make_unique<llvm::raw_fd_ostream>(
      unwrap(outputPath), error, llvm::sys::fs::OF_Text);
  if (error)
    return false;

  mlir::remark::RemarkCategories categories;
  // MLIR only constructs a per-kind filter when that kind is populated; using
  // `all` alone does not activate passed or missed remarks. Limit both kinds
  // directly so unrelated pass remarks never reach this enabled-only path.
  categories.passed = "TokenReuse";
  categories.missed = "TokenReuse";
  auto streamer = std::make_unique<TokenReuseJsonStreamer>(std::move(output));
  auto policy = std::make_unique<mlir::remark::RemarkEmittingPolicyAll>();
  return mlir::succeeded(mlir::remark::enableOptimizationRemarks(
      *ctx, std::move(streamer), std::move(policy), categories));
}
