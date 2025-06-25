#include "catch2/catch.hpp"
#include <tadah/mlip/structure_db.h>
#include "tadah/mlip/dataset_readers/castep_md_reader.h"
#include "tadah/mlip/dataset_readers/castep_geom_reader.h"
#include "tadah/mlip/dataset_readers/castep_castep_reader.h"
#include <tadah/mlip/dataset_readers/vasp_outcar_reader.h>
#include <tadah/mlip/dataset_readers/vasp_vasprun_reader.h>
#include <filesystem>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace tadah::core;
using namespace tadah::mlip;

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
TEST_CASE("Dataset Readers PeriodicTable Initialization") {
    PeriodicTable::initialize();
}

TEST_CASE("Dataset Readers process datasets in directories", "[DatasetReaders]") {

    std::string valid_outcar_dir = "./tests_data/valid_outcars";
    std::string valid_vasprun_dir = "./tests_data/valid_vaspruns";
    std::string valid_castep_md_dir = "./tests_data/valid_castep_md";
    std::string valid_castep_geom_dir = "./tests_data/valid_castep_geom";
    std::string valid_castep_castep_dir = "./tests_data/valid_castep_castep";

    std::vector<std::string> valid_outcar_files = get_all_files(valid_outcar_dir);
    std::vector<std::string> valid_vasprun_files = get_all_files(valid_vasprun_dir);
    std::vector<std::string> valid_castep_md_files = get_all_files(valid_castep_md_dir);
    std::vector<std::string> valid_castep_geom_files = get_all_files(valid_castep_geom_dir);
    std::vector<std::string> valid_castep_castep_files = get_all_files(valid_castep_castep_dir);

    SECTION("Valid OUTCAR datasets - Constructor 1") {
        for (const auto& filename : valid_outcar_files) {
            StructureDB db;
            REQUIRE_NOTHROW(VaspOutcarReader(db));
            VaspOutcarReader reader(db);
            
            REQUIRE_NOTHROW(reader.read_data(filename));
            REQUIRE_NOTHROW(reader.parse_data());
            REQUIRE_NOTHROW(reader.print_summary());
            // Additional checks to confirm data validity
        }
    }
    SECTION("Valid OUTCAR datasets - Constructor 2") {
        for (const auto& filename : valid_outcar_files) {
            StructureDB db;
            REQUIRE_NOTHROW(VaspOutcarReader(db, filename));
            VaspOutcarReader reader(db, filename);
            
            REQUIRE_NOTHROW(reader.print_summary());
            // Additional checks to confirm data validity
        }
    }

    SECTION("Valid vasprun.xml datasets - Constructor 1") {
        for (const auto& filename : valid_vasprun_files) {
            StructureDB db;
            REQUIRE_NOTHROW(VaspVasprunReader(db));
            VaspVasprunReader reader(db);
            
            REQUIRE_NOTHROW(reader.read_data(filename));
            REQUIRE_NOTHROW(reader.parse_data());
            REQUIRE_NOTHROW(reader.print_summary());
            // Additional checks to confirm data validity
        }
    }

    SECTION("Valid vasprun.xml datasets - Constructor 2") {
        for (const auto& filename : valid_vasprun_files) {
            StructureDB db;
            REQUIRE_NOTHROW(VaspVasprunReader(db,filename));
            VaspVasprunReader reader(db, filename);
            
            REQUIRE_NOTHROW(reader.print_summary());
            // Additional checks to confirm data validity
        }
    }

    SECTION("Valid CASTEP .md datasets - Constructor 1") {
        for (const auto& filename : valid_castep_md_files) {
            StructureDB db;
            REQUIRE_NOTHROW(CastepMDReader(db));
            CastepMDReader reader(db);
            
            REQUIRE_NOTHROW(reader.read_data(filename));
            REQUIRE_NOTHROW(reader.parse_data());
            REQUIRE_NOTHROW(reader.print_summary());
            // Additional checks to confirm data validity
        }
    }
    SECTION("Valid CASTEP .md datasets - Constructor 2") {
        for (const auto& filename : valid_castep_md_files) {
            StructureDB db;
            REQUIRE_NOTHROW(CastepMDReader(db, filename));
            CastepMDReader reader(db, filename);
            
            REQUIRE_NOTHROW(reader.print_summary());
            // Additional checks to confirm data validity
        }
    }

    SECTION("Valid CASTEP .geom datasets - Constructor 1") {
        for (const auto& filename : valid_castep_geom_files) {
            StructureDB db;
            REQUIRE_NOTHROW(CastepGeomReader(db));
            CastepGeomReader reader(db);
            
            REQUIRE_NOTHROW(reader.read_data(filename));
            REQUIRE_NOTHROW(reader.parse_data());
            REQUIRE_NOTHROW(reader.print_summary());
            // Additional checks to confirm data validity
        }
    }
    SECTION("Valid CASTEP .geom datasets - Constructor 2") {
        for (const auto& filename : valid_castep_geom_files) {
            StructureDB db;
            REQUIRE_NOTHROW(CastepGeomReader(db, filename));
            CastepGeomReader reader(db, filename);
            
            REQUIRE_NOTHROW(reader.print_summary());
            // Additional checks to confirm data validity
        }
    }

    SECTION("Valid CASTEP .castep datasets - Constructor 1") {
        for (const auto& filename : valid_castep_castep_files) {
            StructureDB db;
            REQUIRE_NOTHROW(CastepCastepReader(db));
            CastepCastepReader reader(db);
            
            REQUIRE_NOTHROW(reader.read_data(filename));
            REQUIRE_NOTHROW(reader.parse_data());
            REQUIRE_NOTHROW(reader.print_summary());
            // Additional checks to confirm data validity
        }
    }
    SECTION("Valid CASTEP .castep datasets - Constructor 2") {
        for (const auto& filename : valid_castep_castep_files) {
            StructureDB db;
            REQUIRE_NOTHROW(CastepCastepReader(db, filename));
            CastepCastepReader reader(db, filename);
            
            REQUIRE_NOTHROW(reader.print_summary());
            // Additional checks to confirm data validity
        }
    }
}
