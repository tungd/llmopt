open Test_support

let () =

  Test_generation.run ();

  Test_schedule.run ();

  Test_hardware.run ();

  Test_model_program.run ();

  Test_cache.run ();

  Test_quant.run ();

  Test_graph.run ();
  Test_metal.run ();

  Test_fold_catalog.run ();

  print_endline "llmopt canonical W4A16/KVQ8 tests passed"
