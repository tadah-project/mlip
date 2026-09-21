#include "catch2/catch.hpp"
#include <tadah/mlip/analytics/analytics.h>
#include <tadah/mlip/analytics/analytics_grad.h>

TEST_CASE("Dataset RMSE partials match finite differences and averaged helpers", "[analytics][pc_dataset]") {
    using namespace tadah::mlip;
    StructureDB truth;
    truth.dbidx = {0, 1, 3}; // deliberately unequal dataset sizes
    truth.structures.resize(3);
    for (std::size_t i = 0; i < 3; ++i) {
        truth(i).energy = 0.3 * (i + 1);
        truth(i).atoms.resize(i + 1); // deliberately unequal atom counts
        for (auto& a : truth(i).atoms) a.force = tadah::core::Vec3d(0.1, 0.2, 0.3);
    }
    auto pred = truth;
    for (std::size_t i = 0; i < 3; ++i) {
        pred(i).energy += 0.01 * (i + 1);
        for (auto& a : pred(i).atoms) a.force += tadah::core::Vec3d(0.04, 0.06, 0.08);
    }
    for (bool forces : {false, true}) {
        INFO("forces=" << forces);
        std::size_t callbacks = 0;
        auto check = [&](std::size_t dataset, std::size_t i, PredCellKind kind,
                         std::size_t comp, double partial) {
            ++callbacks;
            CHECK(dataset == (i == 0 ? 0 : 1));
            CHECK(kind == (forces ? PredCellKind::FORCE : PredCellKind::ENERGY));
            const double h = 1e-7;
            auto plus = pred, minus = pred;
            if (forces) {
                plus(i).atoms[comp/3].force(comp%3) += h;
                minus(i).atoms[comp/3].force(comp%3) -= h;
            } else {
                plus(i).energy += h;
                minus(i).energy -= h;
            }
            Analytics ap(truth, plus), am(truth, minus);
            const auto vp = forces ? ap.calc_f_rmse() : ap.calc_e_rmse();
            const auto vm = forces ? am.calc_f_rmse() : am.calc_e_rmse();
            const double scale = forces ? 1 : 1000;
            CHECK(partial == Approx(scale*(vp[dataset]-vm[dataset])/(2*h)).epsilon(1e-6));
        };
        if (forces) analytics_grad::frmse_per_dataset(truth, pred, 1, check);
        else analytics_grad::ermse_per_dataset(truth, pred, 1000, check);
        CHECK(callbacks == (forces ? 18 : 3));
        double sum = 0, avg = 0;
        auto collect = [&](std::size_t, std::size_t, PredCellKind, std::size_t, double x) { sum += x; };
        auto collect_avg = [&](std::size_t, PredCellKind, std::size_t, double x) { avg += x; };
        if (forces) {
            analytics_grad::frmse_per_dataset(truth, pred, 1, collect);
            analytics_grad::frmse(truth, pred, 1, collect_avg);
        } else {
            analytics_grad::ermse_per_dataset(truth, pred, 1000, collect);
            analytics_grad::ermse(truth, pred, 1000, collect_avg);
        }
        CHECK(avg == Approx(sum/2));
    }
    std::size_t calls = 0;
    auto count = [&](std::size_t, std::size_t, PredCellKind, std::size_t, double) { ++calls; };
    analytics_grad::ermse_per_dataset(truth, truth, 1000, count);
    analytics_grad::frmse_per_dataset(truth, truth, 1, count);
    CHECK(calls == 0); // zero subgradient at perfect fit
    pred.dbidx = {0,3};
    CHECK_THROWS(analytics_grad::ermse_per_dataset(truth, pred, 1000, count));
    CHECK_THROWS(analytics_grad::frmse_per_dataset(truth, pred, 1, count));
}
