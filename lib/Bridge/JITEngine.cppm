//===-- JITEngine.cppm - Reussir JIT engine ---------------------*- c++ -*-===//
//
// Part of the Reussir project, dual licensed under the Apache License v2.0 or
// the MIT License.
// SPDX-License-Identifier: Apache-2.0 OR MIT
//
//===----------------------------------------------------------------------===//
//
// This file implements the JIT engine for Reussir.
//===----------------------------------------------------------------------===//
module;
#include <llvm/ADT/StringRef.h>
#include <llvm/ExecutionEngine/Orc/AbsoluteSymbols.h>
#include <llvm/ExecutionEngine/Orc/CompileOnDemandLayer.h>
#include <llvm/ExecutionEngine/Orc/CompileUtils.h>
#include <llvm/ExecutionEngine/Orc/Core.h>
#include <llvm/ExecutionEngine/Orc/EPCIndirectionUtils.h>
#include <llvm/ExecutionEngine/Orc/ExecutionUtils.h>
#include <llvm/ExecutionEngine/Orc/ExecutorProcessControl.h>
#include <llvm/ExecutionEngine/Orc/IRCompileLayer.h>
#include <llvm/ExecutionEngine/Orc/IRTransformLayer.h>
#include <llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h>
#include <llvm/ExecutionEngine/Orc/MaterializationUnit.h>
#include <llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h>
#include <llvm/ExecutionEngine/Orc/SelfExecutorProcessControl.h>
#include <llvm/ExecutionEngine/Orc/Shared/ExecutorSymbolDef.h>
#include <llvm/ExecutionEngine/Orc/ThreadSafeModule.h>
#include <llvm/ExecutionEngine/SectionMemoryManager.h>
#include <llvm/IR/DataLayout.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/Support/DynamicLibrary.h>
#include <llvm/Support/TargetSelect.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
#include <optional>
#include <spdlog/spdlog.h>

#ifdef REUSSIR_HAS_TPDE
#include <tpde-llvm/OrcCompiler.hpp>
#endif

#include "Reussir/Bridge.h"

export module Reussir.JITEngine;
import Reussir.Bridge;

#ifdef _WIN32
#define REUSSIR_RT_LIBRARY "reussir_rt.dll"
#elif defined(__APPLE__)
#define REUSSIR_RT_LIBRARY "libreussir_rt.dylib"
#else
#define REUSSIR_RT_LIBRARY "libreussir_rt.so"
#endif

namespace reussir {
namespace {
using namespace llvm;
using namespace llvm::orc;
class ReussirASTLayer;

constexpr std::array<const char *, 12> REUSSIR_RT_LIBRARY_HINTS = {
    REUSSIR_RT_LIBRARY,
    "build/lib/" REUSSIR_RT_LIBRARY,
    "./lib" REUSSIR_RT_LIBRARY,
    "../lib" REUSSIR_RT_LIBRARY,
    "../../lib" REUSSIR_RT_LIBRARY,
    "../../../../lib" REUSSIR_RT_LIBRARY,
    "build/bin/" REUSSIR_RT_LIBRARY,
    "./bin/" REUSSIR_RT_LIBRARY,
    "../bin/" REUSSIR_RT_LIBRARY,
    "../../bin/" REUSSIR_RT_LIBRARY,
    "../../../bin/" REUSSIR_RT_LIBRARY,
    "/usr/lib/" REUSSIR_RT_LIBRARY,
};

const char *lookupReussirRTLibrary() {
  if (const char *envPath = std::getenv("REUSSIR_RT_LIBRARY_PATH"))
    return envPath;
  for (const char *hint : REUSSIR_RT_LIBRARY_HINTS)
    if (llvm::sys::fs::exists(hint)) {
      return hint;
    }
  return REUSSIR_RT_LIBRARY;
}

// Holding a pointer to the AST.
class ReussirASTMaterializationUnit final : public MaterializationUnit {
public:
  ReussirASTMaterializationUnit(ReussirASTLayer &ast_layer, ASTStablePtr ast,
                                ArrayRef<char *> symbol_names,
                                ArrayRef<uint8_t> symbol_flags);

  void
  materialize(std::unique_ptr<MaterializationResponsibility> resp) override;

  StringRef getName() const override { return "ReussirJITMaterializationUnit"; }

  ~ReussirASTMaterializationUnit() override;

private:
  ReussirASTLayer &ast_layer;
  ASTStablePtr ast;

  void discard([[maybe_unused]] const JITDylib &dylib,
               [[maybe_unused]] const SymbolStringPtr &Sym) override {
    llvm::report_fatal_error("discard is not supported yet");
  }
};

class ReussirASTLayer final {
public:
  ReussirASTLayer(ASTCallbackFn ast_callback_fn, ASTFreeFn ast_free_fn,
                  IRLayer &inner_layer, llvm::orc::ThreadSafeContext ts_context,
                  mlir::MLIRContext &context, mlir::PassManager &pm,
                  const DataLayout &data_layout, std::string target_triple)
      : ast_callback_fn(ast_callback_fn), ast_free_fn(ast_free_fn),
        inner_layer(inner_layer), ts_context(ts_context), context(context),
        pm(pm), data_layout(data_layout),
        target_triple(std::move(target_triple)) {}

  Error add(ResourceTrackerSP resource_tracker, ASTStablePtr ast,
            ArrayRef<char *> symbol_names, ArrayRef<uint8_t> symbol_flags) {
    return resource_tracker->getJITDylib().define(
        std::make_unique<ReussirASTMaterializationUnit>(
            *this, ast, symbol_names, symbol_flags));
  };

  void emit(std::unique_ptr<MaterializationResponsibility> resp,
            ASTStablePtr ast) {
    auto texture = ast_callback_fn(ast);
    emit(std::move(resp), texture);
    free(const_cast<char *>(texture));
  }

  void emit(std::unique_ptr<MaterializationResponsibility> resp,
            const char *texture) {
    auto unique_module = ts_context.withContextDo([&](LLVMContext *llvmCtx) {
      return bridge::translateToModule(texture, *llvmCtx, context, pm,
                                       data_layout, target_triple);
    });
    auto concurrentModule =
        ThreadSafeModule(std::move(unique_module), ts_context);
    inner_layer.emit(std::move(resp), std::move(concurrentModule));
  }

  MaterializationUnit::Interface
  getInterface(ASTStablePtr ast, llvm::ArrayRef<char *> symbol_names,
               llvm::ArrayRef<uint8_t> symbol_flags) {
    MangleAndInterner mangle(inner_layer.getExecutionSession(), data_layout);
    SymbolFlagsMap symbols;
    for (size_t i = 0; i < symbol_names.size(); ++i) {
      symbols[mangle(symbol_names[i])] = JITSymbolFlags(
          static_cast<JITSymbolFlags::FlagNames>(symbol_flags[i]));
    }
    return MaterializationUnit::Interface(std::move(symbols), nullptr);
  }

private:
  ASTCallbackFn ast_callback_fn;
  ASTFreeFn ast_free_fn;
  IRLayer &inner_layer;
  llvm::orc::ThreadSafeContext ts_context;
  mlir::MLIRContext &context;
  mlir::PassManager &pm;
  const DataLayout &data_layout;
  std::string target_triple;

  friend class ReussirASTMaterializationUnit;
};

ReussirASTMaterializationUnit::~ReussirASTMaterializationUnit() {
  if (ast != nullptr)
    ast_layer.ast_free_fn(ast);
  ast = nullptr;
}

void ReussirASTMaterializationUnit::materialize(
    std::unique_ptr<MaterializationResponsibility> resp) {
  ast_layer.emit(std::move(resp), ast);
  ast = nullptr;
}

ReussirASTMaterializationUnit::ReussirASTMaterializationUnit(
    ReussirASTLayer &ast_layer, ASTStablePtr ast, ArrayRef<char *> symbol_names,
    ArrayRef<uint8_t> symbol_flags)
    : MaterializationUnit(
          ast_layer.getInterface(ast, symbol_names, symbol_flags)),
      ast_layer(ast_layer), ast(ast) {}

IRCompileLayer createCompilerLayer(llvm::orc::JITTargetMachineBuilder jtmb,
                                   ReussirOptOption opt, ExecutionSession &es,
                                   ObjectLayer &base) {
  if (opt == REUSSIR_OPT_TPDE) {
#ifdef REUSSIR_HAS_TPDE
    return {
        es, base,
        std::make_unique<tpde_llvm::ConcurrentOrcCompiler>(std::move(jtmb))};
#else
    spdlog::warn("TPDE is not supported on this platform");
#endif
  }
  return {es, base, std::make_unique<llvm::orc::ConcurrentIRCompiler>(jtmb)};
}

class JITEngine final {
private:
  std::unique_ptr<ExecutionSession> execution_session;
  std::unique_ptr<EPCIndirectionUtils> epc_indirection_utils;

  DataLayout data_layout;
  MangleAndInterner mangle_and_interner;

  RTDyldObjectLinkingLayer object_layer;
  IRCompileLayer compile_layer;
  IRTransformLayer optimize_layer;
  std::unique_ptr<mlir::MLIRContext> context;
  std::unique_ptr<mlir::PassManager> pm;
  llvm::orc::ThreadSafeContext ts_context;
  ReussirASTLayer ast_layer;

  JITDylib &main_dynlib;

  static Expected<ThreadSafeModule>
  optimizeModule(ThreadSafeModule tsm,
                 [[maybe_unused]] const MaterializationResponsibility &resp,
                 ReussirOptOption opt) {
    tsm.withModuleDo([&](Module &m) { bridge::runNPMOptimization(m, opt); });
    return std::move(tsm);
  }

public:
  JITEngine(std::unique_ptr<ExecutionSession> es,
            std::unique_ptr<EPCIndirectionUtils> epciu,
            JITTargetMachineBuilder jtmb, DataLayout dl, ReussirOptOption opt,
            ASTCallbackFn ast_callback_fn, ASTFreeFn ast_free_fn)
      : execution_session(std::move(es)),
        epc_indirection_utils(std::move(epciu)), data_layout(dl),
        mangle_and_interner(*execution_session, dl),
        object_layer(*execution_session,
                     [](const MemoryBuffer &) {
                       return std::make_unique<SectionMemoryManager>();
                     }),
        compile_layer(createCompilerLayer(std::move(jtmb), opt,
                                          *execution_session, object_layer)),
        optimize_layer(
            *execution_session, compile_layer,
            [opt](ThreadSafeModule tsm, MaterializationResponsibility &resp) {
              return optimizeModule(std::move(tsm), resp, opt);
            }),
        context(bridge::buildMLIRContext()),
        pm(bridge::buildPassManager(*context, opt)),
        ts_context(std::make_unique<llvm::LLVMContext>()),
        ast_layer(ast_callback_fn, ast_free_fn, optimize_layer, ts_context,
                  *context, *pm, data_layout,
                  execution_session->getExecutorProcessControl()
                      .getTargetTriple()
                      .str()),
        main_dynlib(execution_session->createBareJITDylib("<main>")) {
    main_dynlib.addGenerator(
        cantFail(DynamicLibrarySearchGenerator::GetForCurrentProcess(
            data_layout.getGlobalPrefix())));
    auto loaded = DynamicLibrarySearchGenerator::Load(
        lookupReussirRTLibrary(), data_layout.getGlobalPrefix());
    if (!loaded)
      spdlog::warn("Failed to load Reussir runtime library");
    else
      main_dynlib.addGenerator(std::move(*loaded));

#if defined(_WIN32) && defined(__x86_64__)
    // On Windows x86-64 (MinGW/Clang64), ___chkstk_ms is a stack-probing
    // intrinsic that LLVM-generated code may call for large stack frames.
    // It comes from compiler-rt builtins (a static library), so it is linked
    // into the host process but not exported by any DLL — meaning
    // GetForCurrentProcess() cannot discover it. We register it manually.
    if (auto addr = llvm::sys::DynamicLibrary::SearchForAddressOfSymbol(
            "___chkstk_ms")) {
      SymbolMap win_symbols;
      win_symbols[mangle_and_interner("___chkstk_ms")] = {
          ExecutorAddr::fromPtr(addr), JITSymbolFlags::Exported};
      cantFail(main_dynlib.define(absoluteSymbols(std::move(win_symbols))));
    }
#endif
  }

  ~JITEngine() {
    if (auto err = execution_session->endSession())
      execution_session->reportError(std::move(err));
    if (auto err = epc_indirection_utils->cleanup())
      execution_session->reportError(std::move(err));
  }

  const DataLayout &getDataLayout() const { return data_layout; }
  JITDylib &getMainJITDylib() { return main_dynlib; }
  Error addModule(const char *texture,
                  ResourceTrackerSP resource_tracker = nullptr) {
    if (!resource_tracker)
      resource_tracker = main_dynlib.getDefaultResourceTracker();

    auto unique_module = ts_context.withContextDo([&](LLVMContext *llvmCtx) {
      return bridge::translateToModule(
          texture, *llvmCtx, *context, *pm, data_layout,
          execution_session->getExecutorProcessControl()
              .getTargetTriple()
              .str());
    });
    auto concurrentModule =
        ThreadSafeModule(std::move(unique_module), ts_context);
    return optimize_layer.add(resource_tracker, std::move(concurrentModule));
  }
  Error addModule(ASTStablePtr ast, char *symbol_names[],
                  uint8_t symbol_flags[], size_t symbol_count,
                  ResourceTrackerSP resource_tracker = nullptr) {
    if (!resource_tracker)
      resource_tracker = main_dynlib.getDefaultResourceTracker();
    llvm::ArrayRef<char *> raw_names(symbol_names, symbol_count);
    llvm::ArrayRef<uint8_t> raw_flags(symbol_flags, symbol_count);
    return ast_layer.add(resource_tracker, ast, raw_names, raw_flags);
  }
  Expected<ExecutorSymbolDef> lookup(StringRef Name, bool mangled) {
    if (mangled) {
      return execution_session->lookup({&main_dynlib},
                                       mangle_and_interner(Name.str()));
    }
    // For unmangled lookups, we need to prepend the global prefix (e.g., '_' on
    // macOS)
    char prefix = data_layout.getGlobalPrefix();
    std::string prefixed_name =
        (prefix != '\0') ? std::string(1, prefix) + Name.str() : Name.str();
    return execution_session->lookup({&main_dynlib}, prefixed_name);
  }
};
} // namespace
} // namespace reussir

static void handleLazyCallThroughError() {
  llvm::report_fatal_error("Failed to handle lazy call through");
}

export extern "C" {
  ReussirJIT reussir_bridge_jit_create(
      ASTCallbackFn ast_callback_fn, ASTFreeFn ast_free_fn,
      ReussirOptOption opt, ReussirLogLevel level) {
    reussir::bridge::setup();
    reussir::bridge::setSpdlogLevel(level);
    using namespace llvm::orc;
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    llvm::InitializeNativeTargetAsmParser();
    auto epc = SelfExecutorProcessControl::Create();
    if (!epc) {
      spdlog::error("Failed to create SelfExecutorProcessControl");
      return nullptr;
    }

    auto es = std::make_unique<ExecutionSession>(std::move(*epc));

    auto epciu = EPCIndirectionUtils::Create(*es);
    if (!epciu) {
      spdlog::error("Failed to create EPCIndirectionUtils");
      return nullptr;
    }

    (*epciu)->createLazyCallThroughManager(
        *es, ExecutorAddr::fromPtr(&handleLazyCallThroughError));

    if (setUpInProcessLCTMReentryViaEPCIU(**epciu)) {
      spdlog::error("Failed to set up in process LCTM reentry via EPCIU");
      return nullptr;
    }

    JITTargetMachineBuilder jtmb(
        es->getExecutorProcessControl().getTargetTriple());

    auto dl = jtmb.getDefaultDataLayoutForTarget();
    if (!dl) {
      spdlog::error("Failed to get default data layout for target");
      return nullptr;
    }
    return new reussir::JITEngine(std::move(es), std::move(*epciu),
                                  std::move(jtmb), std::move(*dl), opt,
                                  ast_callback_fn, ast_free_fn);
  }
  void reussir_bridge_jit_destroy(ReussirJIT jit) {
    delete static_cast<reussir::JITEngine *>(jit);
  }
  bool reussir_bridge_jit_add_lazy_module(
      ReussirJIT jit, ASTStablePtr ast, char *symbol_names[],
      uint8_t symbol_flags[], size_t symbol_count) {
    if (static_cast<reussir::JITEngine *>(jit)->addModule(
            ast, symbol_names, symbol_flags, symbol_count)) {
      spdlog::error("Failed to add lazy module");
      return false;
    }
    return true;
  }
  bool reussir_bridge_jit_add_module(ReussirJIT jit, const char *texture) {
    if (static_cast<reussir::JITEngine *>(jit)->addModule(texture)) {
      spdlog::error("Failed to add module");
      return false;
    }
    return true;
  }
  void *reussir_bridge_jit_lookup_symbol(
      ReussirJIT jit, const char *symbol_name, bool mangled) {
    auto def =
        static_cast<reussir::JITEngine *>(jit)->lookup(symbol_name, mangled);
    if (!def) {
      spdlog::error("Failed to lookup symbol: {}", symbol_name);
      return nullptr;
    }
    return def->getAddress().toPtr<void *>();
  }
  void reussir_bridge_call_f32_func(void *func_ptr, float *result) {
    using F32FuncType = void (*)(float *);
    auto func = reinterpret_cast<F32FuncType>(func_ptr);
    func(result);
  }
  void reussir_bridge_call_f64_func(void *func_ptr, double *result) {
    using F64FuncType = void (*)(double *);
    auto func = reinterpret_cast<F64FuncType>(func_ptr);
    func(result);
  }
  void reussir_bridge_call_str_func(void *func_ptr, ReussirStrResult *result) {
    using StrFuncType = void (*)(ReussirStrResult *);
    auto func = reinterpret_cast<StrFuncType>(func_ptr);
    func(result);
  }
}
