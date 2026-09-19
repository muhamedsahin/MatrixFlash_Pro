#include "matrix_pro/matrix_pro.hpp"
#include "../support/test_support.hpp"

#include <cstdio>
#include <string>

using namespace matrix_pro;
using matrix_pro::test::Context;

int main() {
    try {
        Context ctx;

        const std::string tensor_file = "test_tensor_tmp.mft";
        const std::string checkpoint_file = "test_checkpoint_tmp.mfc";

        ctx.section("save_tensor & load_tensor roundtrip");
        {
            Tensor orig({2, 3, 4});
            orig.fill(7.5f);
            save_tensor(orig, tensor_file);

            Tensor loaded = load_tensor(tensor_file);
            ctx.check(loaded.rank() == 3, "loaded rank = 3");
            ctx.check(loaded.shape() == std::vector<std::size_t>({2, 3, 4}), "loaded shape matches");
            ctx.check(loaded.size() == 24, "loaded size = 24");
            loaded.download();
            bool all_match = true;
            for (float v : loaded.data()) {
                if (v != 7.5f) { all_match = false; break; }
            }
            ctx.check(all_match, "loaded tensor values match saved values");
            std::remove(tensor_file.c_str());
        }

        ctx.section("Checkpoint multi-entry save & load");
        {
            Checkpoint ckpt;
            Matrix weights(2, 3, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
            Tensor biases({3}, {0.1f, 0.2f, 0.3f});

            ckpt.add("weights", weights);
            ckpt.add("biases", biases);
            ckpt.set_metadata("epoch", "42");
            ckpt.set_metadata("arch", "mlp");

            ctx.check(ckpt.contains("weights"), "contains weights");
            ctx.check(ckpt.contains("biases"), "contains biases");
            ctx.check(ckpt.size() == 2, "size is 2");

            ckpt.save(checkpoint_file);

            // Load back
            Checkpoint loaded_ckpt = Checkpoint::load(checkpoint_file);
            ctx.check(loaded_ckpt.contains("weights"), "loaded contains weights");
            ctx.check(loaded_ckpt.contains("biases"), "loaded contains biases");
            ctx.check(loaded_ckpt.get_metadata("epoch") == "42", "metadata epoch matches");
            ctx.check(loaded_ckpt.get_metadata("arch") == "mlp", "metadata arch matches");

            Matrix loaded_w = loaded_ckpt.get_matrix("weights");
            Tensor loaded_b = loaded_ckpt.get_tensor("biases");

            loaded_w.download();
            loaded_b.download();

            ctx.check(loaded_w.rows() == 2 && loaded_w.cols() == 3, "loaded matrix dims match");
            ctx.check_near(loaded_w.at(0, 0), 1.0f, 1e-5, "weights(0,0)");
            ctx.check_near(loaded_w.at(1, 2), 6.0f, 1e-5, "weights(1,2)");

            ctx.check(loaded_b.rank() == 1 && loaded_b.size() == 3, "loaded bias shape matches");
            ctx.check_near(loaded_b.data()[0], 0.1f, 1e-5, "bias[0]");
            ctx.check_near(loaded_b.data()[2], 0.3f, 1e-5, "bias[2]");

            std::remove(checkpoint_file.c_str());
        }

        return ctx.summary("SERIALIZATION");
    } catch (const std::exception& e) {
        matrix_pro::test::Context{}.fatal(e);
        return 99;
    }
}

