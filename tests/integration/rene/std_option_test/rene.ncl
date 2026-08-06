{
  package = { name = "std_option_test", version = "0.1.0" },
  dependencies = {
    std = { path = "../../../../library/std", version = "^0.1" },
    test_utils = { path = "../../../../library/test_utils", version = "^0.1" },
  },
  targets.std_option_test = { kind = 'executable },
}
