#include <tadah/core/logger.h>
#include <tadah/mlip/engines/data_convert_engine.h>
#include <tadah/mlip/structure_db.h>
#include <tadah/mlip/dataset_readers/dataset_reader_selector.h>


#include <iostream>

namespace tadah {
namespace mlip {
namespace engines {

int DataConvertEngine::run(tadah::mlip::Task &task) {
  try {
    core::Context taskContext = task.getContext();

    core::Logger::instance().log(tadah::core::LogLevel::INFO,
        "DataConvertEngine started...");
    StructureDB stdb;

    std::vector<std::string> datasets;
    if (taskContext.contains("DFT-FILE")) {
      datasets.resize(taskContext.size("DFT-FILE"));
      taskContext.get<std::vector<std::string>>("DFT-FILE", datasets);
    } else {
      throw std::runtime_error("No dataset provided.");
    }
    std::vector<std::string> summaryStrVec;
    for (auto &ds: datasets) {
      StructureDB stdb_;
      std::unique_ptr<DatasetReader> reader = DatasetReaderSelector::get_reader(ds,stdb_);
      reader->parse_data();
      if (reader)
        summaryStrVec.push_back(reader->get_summary());
      stdb.add(stdb_);
    }

    std::string outFile = taskContext.get<std::string>("OUTFILE");
    bool append = taskContext.get<bool>("APPEND");
    size_t numeric = StructureDB::dataset_digits(taskContext);
    stdb.dump_to_file(outFile,append, numeric);
    if (core::Logger::instance().getLogLevel() == core::LogLevel::INFO) {
      int count = 0;
      for (const auto &s : summaryStrVec) {
        std::cout << "Set: " << std::to_string(count+1) << " | File: " << datasets[count] << std::endl;
        std::cout<< "   " << s;
        count++;
      }
      std::cout << "-----------------------------------------" << std::endl;
      std::cout << "|                SUMMARY                |" << std::endl;
      std::cout << "-----------------------------------------" << std::endl;
      std::cout << "Total number of sets: " << summaryStrVec.size() << std::endl;
      std::cout << "Total number of configurations: " << stdb.size() << std::endl;
      std::cout << "Total number of atoms: " << stdb.calc_natoms() << std::endl;
      std::cout << "Elements: ";
      for (const auto &e: stdb.get_unique_elements()) {
        std::cout << e.symbol << " ";
      }
      std::cout << std::endl;
      std::cout << "Dataset dumped to: " << outFile << std::endl;
    }

  } catch (const std::exception &e) {
    std::cerr << "DataConvertEngine error: " << e.what() << std::endl;
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
void DataConvertEngine::validate(const tadah::mlip::Task &) {
  core::Logger::instance().log(tadah::core::LogLevel::INFO,
                             "Validating DataConvertEngine...");
}


} // namespace engines
} // namespace mlip
} // namespace tadah
