add_executable(dune-mgis-tensile-test dune-mgis-tensile-test.cc)
target_link_dune_default_libraries(dune-mgis-tensile-test)
target_link_libraries(dune-mgis-tensile-test
	  mgis::MFrontGenericInterface)

add_executable(dune-mgis-ssna303 dune-mgis-ssna303.cc)
target_link_dune_default_libraries(dune-mgis-ssna303)
target_link_libraries(dune-mgis-ssna303
	  mgis::MFrontGenericInterface)
