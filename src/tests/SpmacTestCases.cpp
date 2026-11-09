#include <memory>
#include <vector>

#include "gtest/gtest.h"
#include "tests/PIMKernel.h"

using namespace DRAMSim;

TEST(SpmacIsa, encodingRoundTrip)
{
    PIMCmd in(PIMCmdType::SPMAC, PIMOpdType::GRF_B, PIMOpdType::ODD_BANK, PIMOpdType::GRF_A, 0, 3);
    PIMCmd out;
    out.fromInt(in.toInt());
    EXPECT_TRUE(out.type_ == PIMCmdType::SPMAC);
    EXPECT_TRUE(out.dst_ == PIMOpdType::GRF_B);
    EXPECT_TRUE(out.src0_ == PIMOpdType::ODD_BANK);
    EXPECT_TRUE(out.src1_ == PIMOpdType::GRF_A);
    EXPECT_EQ(out.dstIdx_, 3);

    PIMCmd ld(PIMCmdType::LDIDX, PIMOpdType::A_OUT, PIMOpdType::EVEN_BANK);
    PIMCmd ld_out;
    ld_out.fromInt(ld.toInt());
    EXPECT_TRUE(ld_out.type_ == PIMCmdType::LDIDX);
    EXPECT_TRUE(ld_out.src0_ == PIMOpdType::EVEN_BANK);
}

TEST(SpmacKernel, gathersPerBlockAndAccumulates)
{
    const int num_pairs = 2, data_row = 0, result_row = 16;
    auto mem = make_shared<MultiChannelMemorySystem>("ini/HBM2_samsung_2M_16B_x64.ini",
                                                     "system_hbm_1ch.ini", ".", "example_app",
                                                     256 * 2);
    auto kernel = make_shared<PIMKernel>(mem, 1, 1);
    const int num_blocks = getConfigParam(UINT, "NUM_PIM_BLOCKS");

    // x[j] = j, stored across GRF_A[0..7]
    vector<BurstType> x(8);
    for (int j = 0; j < 128; j++) x[j / 16].fp16Data_[j % 16] = convertF2H((float)j);

    // Layout per even bank: col 0 = idx0, col 1 = val0, col 2 = idx1, col 3 = val1
    vector<BurstType> preload(num_blocks * 2 * num_pairs);
    vector<float> expected(num_blocks * 16, 0.0f);
    for (int pb = 0; pb < num_blocks; pb++)
    {
        int bank = pb * 2;
        int bg = bank / 4, ba = bank % 4;
        for (int k = 0; k < num_pairs; k++)
        {
            BurstType& idx = preload[(pb * num_pairs + k) * 2];
            BurstType& val = preload[(pb * num_pairs + k) * 2 + 1];
            for (int lane = 0; lane < 16; lane++)
            {
                int j = (lane * 7 + pb * 13 + k * 31) % 128;
                float v = (float)(lane % 4 + 1);
                idx.u16Data_[lane] = j;
                val.fp16Data_[lane] = convertF2H(v);
                expected[pb * 16 + lane] += v * j;
            }
            mem->addTransaction(true, kernel->pim_addr_mgr_->addrGen(0, 0, bg, ba, data_row, 2 * k),
                                &idx);
            mem->addTransaction(
                true, kernel->pim_addr_mgr_->addrGen(0, 0, bg, ba, data_row, 2 * k + 1), &val);
        }
    }

    vector<PIMCmd> cmds{
        PIMCmd(PIMCmdType::LDIDX, PIMOpdType::A_OUT, PIMOpdType::EVEN_BANK),
        PIMCmd(PIMCmdType::SPMAC, PIMOpdType::GRF_B, PIMOpdType::EVEN_BANK, PIMOpdType::GRF_A),
        PIMCmd(PIMCmdType::JUMP, num_pairs - 1, 3),
        PIMCmd(PIMCmdType::NOP, 7),
        PIMCmd(PIMCmdType::EXIT, 0),
    };
    kernel->executeSpmacProgram(cmds, x.data(), data_row, num_pairs, result_row);

    // GRF_B[0] of block pb was flushed to bank pb * 2 + 1
    vector<BurstType> result(num_blocks);
    for (int pb = 0; pb < num_blocks; pb++)
    {
        int bank = pb * 2 + 1;
        mem->addTransaction(
            false, kernel->pim_addr_mgr_->addrGen(0, 0, bank / 4, bank % 4, result_row, 0),
            &result[pb]);
    }
    kernel->runPIM();

    for (int pb = 0; pb < num_blocks; pb++)
        for (int lane = 0; lane < 16; lane++)
            EXPECT_FLOAT_EQ(convertH2F(result[pb].fp16Data_[lane]), expected[pb * 16 + lane])
                << "block " << pb << " lane " << lane;
}