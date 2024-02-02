cmake_minimum_required(VERSION 3.16.3...3.19.7 FATAL_ERROR)

project(ClassificationScore)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://www.slicer.org/wiki/Documentation/Nightly/Extensions/ClassificationScore")
set(EXTENSION_CATEGORY "Classification")
set(EXTENSION_CONTRIBUTORS "Roberto Veraldi (Magna Graecia University of Catanzaro, Italy)", "Paolo Zaffino (Magna Graecia University of Catanzaro, Italy)", "Maria Francesca Spadea (KIT, Germany)")
set(EXTENSION_DESCRIPTION "This extention is usefull for your classification application.  It is going to classify your classification score depending on your data and your pth file.")
set(EXTENSION_ICONURL "https://www.example.com/Slicer/Extensions/ClassificationScore.png")
set(EXTENSION_SCREENSHOTURLS "https://www.example.com/Slicer/Extensions/ClassificationScore/Screenshots/1.png")
set(EXTENSION_DEPENDS "NA") # Specified as a list or "NA" if no dependencies

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(Classificator)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
