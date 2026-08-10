{
  package = { name = "array_hashmap_stress", version = "0.1.0" },
  dependencies = {
    std = { path = "../../../../library/std", version = "^0.1" },
    test_utils = { path = "../../../../library/test_utils", version = "^0.1" },
  },
  targets.array_hashmap_stress = { kind = 'executable },
}
