/******************************************************************************\
 * Technische Universitaet Darmstadt, Institut fuer Nachrichtentechnik
 * Copyright (c) 2001
 *
 * Author(s):
 *      Volker Fischer
 *
 * Description:
 *
 * Note:
 * Input data stream is divided into a regluar stream and the VSPP stream.
 * Both stream are treated independently. The respective positions of the
 * two streams are requested in the init-routine.
 *
 ******************************************************************************
 *
 * This program is free software; you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the Free Software
 * Foundation; either version 2 of the License, or (at your option) any later
 * version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the GNU General Public License along with
 * this program; if not, write to the Free Software Foundation, Inc.,
 * 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 *
\******************************************************************************/

#include "EnergyDispersal.h"

#include <fstream>
#include <cstdio>
#include <cstdint>

/* ---- MSC RAW DUMP ---- */
#ifdef _WIN32
static const char* MSC_DUMP_PATH = "D:\\KCBSDRM\\temp\\dream_msc_dump.bin";
#else
static const char* MSC_DUMP_PATH = "/tmp/dream_msc_dump.bin";
#endif
static std::ofstream g_msc_dump;
static bool          g_msc_dump_open = false;
/* ----------------------- */


/* Implementation *************************************************************/
void CEngergyDispersal::ProcessData(CVector<_BINARY>* pbiData)
{
    int                 i;
    uint32_t    iTempShiftRegister;
    _BINARY             biPRBSbit;

    /* Init shift register and set all registers to "1" with bit-wise
       not-operation */
    iShiftRegisterSPP = ~uint32_t(0);
    iShiftRegisterVSPP = ~uint32_t(0);

    /* Main routine */
    for (i = 0; i < iNumInBits; i++)
    {
        if (i < iEndIndVSPP)
        {
            /* Calculate new PRBS bit */
            iTempShiftRegister = iShiftRegisterVSPP;

            /* P(X) = X^9 + X^5 + 1,
               in this implementation we have to shift n-1! */
            biPRBSbit = _BINARY(((iTempShiftRegister >> 4) & 1) ^
                                ((iTempShiftRegister >> 8) & 1));

            /* Shift bits in shift register and add new bit */
            iShiftRegisterVSPP <<= 1;
            iShiftRegisterVSPP |= (biPRBSbit & 1);
        }
        else
        {
            /* Calculate new PRBS bit */
            iTempShiftRegister = iShiftRegisterSPP;

            /* P(X) = X^9 + X^5 + 1,
               in this implementation we have to shift n-1! */
            biPRBSbit = _BINARY(((iTempShiftRegister >> 4) & 1) ^
                                ((iTempShiftRegister >> 8) & 1));

            /* Shift bits in shift register and add new bit */
            iShiftRegisterSPP <<= 1;
            iShiftRegisterSPP |= (biPRBSbit & 1);
        }

        /* Apply PRBS to the data-stream */
        (*pbiData)[i] ^= biPRBSbit;
    }

    /* ---- MSC RAW DUMP (after PRBS de-scrambling) ---- */
    {
        if (!g_msc_dump_open) {
            g_msc_dump.open(MSC_DUMP_PATH, std::ios::binary | std::ios::app);
            if (g_msc_dump.is_open()) {
                g_msc_dump_open = true;
                fprintf(stderr, "[MSC DUMP] Writing to: %s\n", MSC_DUMP_PATH);
            } else {
                fprintf(stderr, "[MSC DUMP] Cannot open %s\n", MSC_DUMP_PATH);
            }
        }
        if (g_msc_dump_open && (*pbiData).Size() > 0) {
            const int _nb = (*pbiData).Size() / 8;
            for (int _i = 0; _i < _nb; _i++) {
                uint8_t _b = 0;
                for (int _j = 0; _j < 8; _j++)
                    _b = (_b << 1) | (uint8_t((*pbiData)[_i*8+_j]) & 1);
                g_msc_dump.put((char)_b);
            }
            g_msc_dump.flush();
        }
    }
    /* ---- END MSC RAW DUMP ---- */
}

void CEngergyDispersal::Init(int iNewNumInBits, int iNewLengthVSPP)
{
    /* Set the internal parameters */
    iNumInBits = iNewNumInBits;
    iEndIndVSPP = iNewLengthVSPP;
}
