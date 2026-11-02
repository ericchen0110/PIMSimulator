/***************************************************************************************************
 * Copyright (C) 2021 Samsung Electronics Co. LTD
 *
 * This software is a property of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed,
 * transmitted, transcribed, stored in a retrieval system, or translated into any human
 * or computer language in any form by any means,electronic, mechanical, manual or otherwise,
 * or disclosed to third parties without the express written permission of Samsung Electronics.
 * (Use of the Software is restricted to non-commercial, personal or academic, research purpose
 * only)
 **************************************************************************************************/

#include "PIMBlock.h"

#include <sstream>
#include <string>

#include "PrintMacros.h"
#include "SystemConfiguration.h"
#include "half.h"

using namespace DRAMSim;

void PIMBlock::add(BurstType& dstBst, BurstType& src0Bst, BurstType& src1Bst)
{
    if (pimPrecision_ == FP16)
    {
        for (int i = 0; i < 16; i++)
        {
            dstBst.fp16Data_[i] = src0Bst.fp16Data_[i] + src1Bst.fp16Data_[i];
        }
    }
    else if (pimPrecision_ == FP32)
    {
        for (int i = 0; i < 8; i++)
        {
            dstBst.fp32Data_[i] = src0Bst.fp32Data_[i] + src1Bst.fp32Data_[i];
        }
    }
    else
        dstBst = src0Bst + src1Bst;
}

void PIMBlock::mul(BurstType& dstBst, BurstType& src0Bst, BurstType& src1Bst)
{
    if (pimPrecision_ == FP16)
    {
        for (int i = 0; i < 16; i++)
        {
            dstBst.fp16Data_[i] = src0Bst.fp16Data_[i] * src1Bst.fp16Data_[i];
        }
    }
    else if (pimPrecision_ == FP32)
    {
        for (int i = 0; i < 8; i++)
        {
            dstBst.fp32Data_[i] = src0Bst.fp32Data_[i] * src1Bst.fp32Data_[i];
        }
    }
    else
        dstBst = src0Bst * src1Bst;
}

void PIMBlock::mac(BurstType& dstBst, BurstType& src0Bst, BurstType& src1Bst)
{
    if (pimPrecision_ == FP16)
    {
        for (int i = 0; i < 16; i++)
        {
            dstBst.fp16Data_[i] = src0Bst.fp16Data_[i] * src1Bst.fp16Data_[i] + dstBst.fp16Data_[i];
        }

        DEBUG("MAC " << src0Bst.hexToStr2() << "*+" << src1Bst.hexToStr2() << ""
                     << dstBst.hexToStr2());
    }
    else if (pimPrecision_ == FP32)
    {
        for (int i = 0; i < 8; i++)
        {
            dstBst.fp32Data_[i] = src0Bst.fp32Data_[i] * src1Bst.fp32Data_[i] + dstBst.fp32Data_[i];
        }
    }
    else
        dstBst = src0Bst * src1Bst + dstBst;
}

void PIMBlock::mad(BurstType& dstBst, BurstType& src0Bst, BurstType& src1Bst, BurstType& src2Bst)
{
    if (pimPrecision_ == FP16)
    {
        for (int i = 0; i < 16; i++)
        {
            dstBst.fp16Data_[i] =
                src0Bst.fp16Data_[i] * src1Bst.fp16Data_[i] + src2Bst.fp16Data_[i];
        }
    }
    else if (pimPrecision_ == FP32)
    {
        for (int i = 0; i < 8; i++)
        {
            dstBst.fp32Data_[i] =
                src0Bst.fp32Data_[i] * src1Bst.fp32Data_[i] + src2Bst.fp32Data_[i];
        }
    }
    else
        dstBst = src0Bst * src1Bst + src2Bst;
}

void PIMBlock::gather(BurstType& dstBst) const
{
    if (pimPrecision_ != FP16)
    {
        ERROR("== Error - SPMAC gather only supports FP16");
        exit(-1);
    }

    for (int lane = 0; lane < 16; lane++)
    {
        unsigned j = idx.u16Data_[lane] & 0x7f;  // 8 registers * 16 lanes = 128 entries
        dstBst.fp16Data_[lane] = grfA[j >> 4].fp16Data_[j & 0xf];
    }
}

std::string PIMBlock::print()
{
    stringstream ss;
    ss << "[SRF]" << srf.binToStr() << endl;
    for (int i = 0; i < 8; i++) ss << "[GRF_A][" << i << "]" << grfA[i].fp16ToStr() << endl;
    for (int i = 0; i < 8; i++) ss << "[GRF_B][" << i << "]" << grfB[i].fp16ToStr() << endl;
    ss << "[M_OUT]" << mOut.binToStr() << endl;
    ss << "[A_OUT]" << aOut.binToStr() << endl;
    ss << "[IDX]" << idx.hexToStr() << endl;

    return ss.str();
}
