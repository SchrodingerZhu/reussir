{
  package = {
    name = "std_integration",
    version = "0.1.0",
  },
  dependencies = {
    std = { path = "../../../library/std", version = "^0.1" },
    test_utils = { path = "../../../library/test_utils", version = "^0.1" },
  },
  targets = {
    cow_hash = { kind = 'executable, path = "cow_hash.rr" },
    cow_ordered = { kind = 'executable, path = "cow_ordered.rr" },
    cow_seq = { kind = 'executable, path = "cow_seq.rr" },
    fingertree = { kind = 'executable, path = "fingertree.rr" },
    hash = { kind = 'executable, path = "hash.rr" },
    hashmap = { kind = 'executable, path = "hashmap.rr" },
    hashset = { kind = 'executable, path = "hashset.rr" },
    heap = { kind = 'executable, path = "heap.rr" },
    list = { kind = 'executable, path = "list.rr" },
    option = { kind = 'executable, path = "option.rr" },
    rtqueue = { kind = 'executable, path = "rtqueue.rr" },
    string = { kind = 'executable, path = "string.rr" },
    wavl = { kind = 'executable, path = "wavl.rr" },
  },
}
