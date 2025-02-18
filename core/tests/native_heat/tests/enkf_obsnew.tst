<?xml version="1.0" encoding="UTF-8"?>
<testConfig>
        <id>/home/verlaanm/deltares/src/openda_20110523/public/tests/native_heat/./enkf_obsnew.oda</id>
        <odaFile>../enkf_obsnew.oda</odaFile>
        <checks>
                <check>
                        <file removeBeforeTest="yes" >../enkf_obsnew.log</file>
                        <find>===DONE===</find>
                </check>
                <check>
                        <file removeBeforeTest="yes" >../enkf_obsnew_results.m</file>
			<!-- last line looks like
                        x_f_central{100}	=[0.8229010069629028,0.7927305990020073
			-->
			<regex>x_f_central\{100\}(.*)0.82290(.*),0.79273(.*)</regex>
                </check>
        </checks>
</testConfig>
