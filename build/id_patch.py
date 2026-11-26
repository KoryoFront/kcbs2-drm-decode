"""id_patch.py -- make Dream log the station's self-identification (FAC + SDC) when KCBS_ID is set."""
import sys
root = sys.argv[1]

SDC_LABEL = r'''
        if (getenv("KCBS_ID")) {
            static std::string last;
            if (strLabel != last) {
                last = strLabel;
                FILE* f = fopen(getenv("KCBS_ID"), "a");
                if (f) {
                    fprintf(f, "SDC label[%d] len %d hex:", iTempShortID, iLengthOfBody);
                    for (unsigned char c : strLabel) fprintf(f, " %02X", c);
                    fprintf(f, "  text: %s\n", strLabel.c_str());
                    fclose(f);
                }
            }
        }
'''

SDC_T12 = r'''
    if (getenv("KCBS_ID")) {
        static int once = 0;
        if (!once++) {
            FILE* f = fopen(getenv("KCBS_ID"), "a");
            if (f) { fprintf(f, "SDC type 12 (language+country) present, short id %d\n", iShortID); fclose(f); }
        }
    }
'''

SDC_T12_AFTER = r'''
    if (getenv("KCBS_ID")) {
        static int once2 = 0;
        if (!once2++) {
            FILE* f = fopen(getenv("KCBS_ID"), "a");
            if (f) { fprintf(f, "SDC type 12: language code '%s' country '%s'\n",
                             Parameter.Service[iShortID].strLanguageCode.c_str(),
                             Parameter.Service[iShortID].strCountryCode.c_str()); fclose(f); }
        }
    }
'''

FAC = r'''
        if (getenv("KCBS_ID")) {
            static int n = 0;
            if (n++ < 3) {
                FILE* f = fopen(getenv("KCBS_ID"), "a");
                if (f) {
                    fprintf(f, "FAC short id %d: service id %06X  language %d  descr/pty %d  audio(0)/data(1) %d\n",
                            iTempShortID, (unsigned)Parameter.Service[iTempShortID].iServiceID,
                            Parameter.Service[iTempShortID].iLanguage, Parameter.Service[iTempShortID].iServiceDescr,
                            (int)Parameter.Service[iTempShortID].eAudDataFlag);
                    fclose(f);
                }
            }
        }
'''

def patch(path, anchor_start, anchor_end_char, code, after=True):
    s = open(path).read()
    if code.strip()[:40] in s:
        return
    i = s.index(anchor_start)
    j = s.index(anchor_end_char, i) + 1
    s = s[:j] + code + s[j:] if after else s[:i] + code + s[i:]
    if "#include <cstdlib>" not in s:
        s = "#include <cstdlib>\n#include <cstdio>\n#include <string>\n" + s
    open(path, "w").write(s)

sdc = root + "/src/SDC/SDCReceive.cpp"
patch(sdc, "Parameter.Service[iTempShortID].strLabel = strLabel;", ";", SDC_LABEL)
patch(sdc, 'Parameter.Service[iShortID].strCountryCode = "";', ";", SDC_T12, after=False)
fac = root + "/src/FAC/FAC.cpp"
patch(fac, "Parameter.Service[iTempShortID].iServiceDescr =", ";", FAC)
print("patched")
