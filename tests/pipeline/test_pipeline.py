import os
from cosipy import test_data
from cosipy.pipeline.task.task import cosi_bindata
from cosipy.pipeline.task.task import cosi_threemlfit

data_path=str(test_data.path)
config_path=os.path.join(data_path,"test_pipeline.yaml")

def test_cosi_bindata(tmp_path):
    tmpdir = tmp_path.as_posix()
    os.system(str("cosi-bindata --config "+ config_path + " -o " + tmpdir + " --overwrite"))
    cosi_bindata(["--config",config_path,"-o",tmpdir,"--overwrite"])

def test_cosi_threemlfit(tmp_path):
    tmpdir = tmp_path.as_posix()
    os.system(str("cosi-threemlfit --config "+ config_path + " -o " + tmpdir + " --overwrite"))
    cosi_threemlfit(["--config",config_path,"-o",tmpdir,"--overwrite"])


