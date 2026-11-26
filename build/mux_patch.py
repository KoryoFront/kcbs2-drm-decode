"""mux_patch.py -- log the SDC multiplex description (streams, lengths, protection) when KCBS_ID is set."""
import sys
p = sys.argv[1] + "/src/SDC/SDCReceive.cpp"
s = open(p).read()
MARK = "KCBS_MUX"
if MARK not in s:
    anchor = "    /* Set new parameters in global struct */\n    Parameter.SetMSCProtLev(MSCPrLe, bWithHierarch);"
    code = r'''    /* KCBS_MUX */
    if (getenv("KCBS_ID")) {
        static int nlog = 0;
        if (nlog++ < 3) {
            FILE* f = fopen(getenv("KCBS_ID"), "a");
            if (f) {
                fprintf(f, "SDC type 0 (multiplex): %d stream(s); protection partA %d partB %d hier %d; MSC coding scheme %d\n",
                        iNumStreams, MSCPrLe.iPartA, MSCPrLe.iPartB, (int)bWithHierarch, (int)Parameter.eMSCCodingScheme);
                for (int k = 0; k < iNumStreams; k++)
                    fprintf(f, "   stream %d: lenPartA %d bytes, lenPartB %d bytes\n", k,
                            Parameter.Stream[k].iLenPartA, Parameter.Stream[k].iLenPartB);
                fclose(f);
            }
        }
    }
'''
    if anchor not in s:
        sys.exit("anchor not found")
    s = s.replace(anchor, code + anchor)
    open(p, "w").write(s)
print("patched")
