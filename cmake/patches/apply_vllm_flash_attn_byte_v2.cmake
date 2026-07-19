# Apply the ByteV2 FA2 patch idempotently. FetchContent's Git update step can
# cause its patch step to run again against a persistent source checkout.

if(NOT DEFINED SOURCE_DIR OR NOT IS_DIRECTORY "${SOURCE_DIR}")
  message(FATAL_ERROR "A valid vllm-flash-attn SOURCE_DIR is required")
endif()
if(NOT DEFINED PATCH_FILE OR NOT EXISTS "${PATCH_FILE}")
  message(FATAL_ERROR "The ByteV2 FA2 PATCH_FILE does not exist")
endif()

find_package(Git REQUIRED)

execute_process(
  COMMAND "${GIT_EXECUTABLE}" apply --reverse --check --whitespace=nowarn
          "${PATCH_FILE}"
  WORKING_DIRECTORY "${SOURCE_DIR}"
  RESULT_VARIABLE reverse_check_result
  OUTPUT_QUIET
  ERROR_QUIET
)
if(reverse_check_result EQUAL 0)
  message(STATUS "ByteV2 FA2 patch is already applied")
  return()
endif()

execute_process(
  COMMAND "${GIT_EXECUTABLE}" apply --check --whitespace=nowarn
          "${PATCH_FILE}"
  WORKING_DIRECTORY "${SOURCE_DIR}"
  RESULT_VARIABLE forward_check_result
  OUTPUT_VARIABLE forward_check_stdout
  ERROR_VARIABLE forward_check_stderr
)
if(NOT forward_check_result EQUAL 0)
  message(FATAL_ERROR
    "ByteV2 FA2 patch is neither cleanly applied nor cleanly applicable.\n"
    "For a managed FetchContent checkout, remove only its vllm-flash-attn "
    "source/subbuild directories and reconfigure.\n"
    "stdout:\n${forward_check_stdout}\n"
    "stderr:\n${forward_check_stderr}"
  )
endif()

execute_process(
  COMMAND "${GIT_EXECUTABLE}" apply --whitespace=nowarn "${PATCH_FILE}"
  WORKING_DIRECTORY "${SOURCE_DIR}"
  RESULT_VARIABLE apply_result
  OUTPUT_VARIABLE apply_stdout
  ERROR_VARIABLE apply_stderr
)
if(NOT apply_result EQUAL 0)
  message(FATAL_ERROR
    "Failed to apply the ByteV2 FA2 patch.\n"
    "stdout:\n${apply_stdout}\n"
    "stderr:\n${apply_stderr}"
  )
endif()

message(STATUS "Applied ByteV2 FA2 patch")
