cmake_minimum_required(VERSION 3.12)

project(Tadah.MLIP)

file(GLOB_RECURSE MLIP_SRC2 CONFIGURE_DEPENDS "*.cpp")
set(MLIP_SRC ${MLIP_SRC2} PARENT_SCOPE)


#add_library(tadah.mlip
#    ${MLIP_SRC}
#    )

