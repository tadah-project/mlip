#include "catch2/catch.hpp"
#include <tadah/mlip/structure_db.h>
#include <tadah/mlip/dataset_readers/outcar_reader.h>
#include <tadah/mlip/dataset_readers/vasprun_reader.h>
#include <filesystem>
#include <string>
#include <vector>

namespace fs = std::filesystem;

// Function to list all files in a directory
std::vector<std::string> get_all_files(const std::string& directory) {
    std::vector<std::string> filenames;
    for (const auto& entry : fs::directory_iterator(directory)) {
        if (entry.is_regular_file()) {
            filenames.push_back(entry.path().string());
        }
    }
    return filenames;
}

TEST_CASE("Dataset Readers process datasets in directories", "[DatasetReaders]") {

    std::string valid_outcar_dir = "./tests_data/valid_outcars";
    std::string valid_vasprun_dir = "./tests_data/valid_vaspruns";

    std::vector<std::string> valid_outcar_files = get_all_files(valid_outcar_dir);
    std::vector<std::string> valid_vasprun_files = get_all_files(valid_vasprun_dir);

    SECTION("Valid OUTCAR datasets - Constructor 1") {
        for (const auto& filename : valid_outcar_files) {
            StructureDB db;
            REQUIRE_NOTHROW(OutcarReader(db));
            OutcarReader reader(db);
            
            REQUIRE_NOTHROW(reader.read_data(filename));
            REQUIRE_NOTHROW(reader.parse_data());
            REQUIRE_NOTHROW(reader.print_summary());
            // Additional checks to confirm data validity
        }
    }
    SECTION("Valid OUTCAR datasets - Constructor 2") {
        for (const auto& filename : valid_outcar_files) {
            StructureDB db;
            REQUIRE_NOTHROW(OutcarReader(db, filename));
            OutcarReader reader(db, filename);
            
            REQUIRE_NOTHROW(reader.print_summary());
            // Additional checks to confirm data validity
        }
    }

    SECTION("Valid vasprun.xml datasets - Constructor 1") {
        for (const auto& filename : valid_vasprun_files) {
            StructureDB db;
            REQUIRE_NOTHROW(VasprunReader(db));
            VasprunReader reader(db);
            
            REQUIRE_NOTHROW(reader.read_data(filename));
            REQUIRE_NOTHROW(reader.parse_data());
            REQUIRE_NOTHROW(reader.print_summary());
            // Additional checks to confirm data validity
        }
    }

    SECTION("Valid vasprun.xml datasets - Constructor 2") {
        for (const auto& filename : valid_vasprun_files) {
            StructureDB db;
            REQUIRE_NOTHROW(VasprunReader(db,filename));
            VasprunReader reader(db, filename);
            
            REQUIRE_NOTHROW(reader.print_summary());
            // Additional checks to confirm data validity
        }
    }
}
