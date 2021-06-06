simply run 

```shell
pip  install --user -i https://pypi.gurobi.com gurobipy
pip  install -f https://download.mosek.com/stable/wheel/index.html Mosek --user
pip install -r requirements.txt
python test.py -config configs/test_mosek.yml
```

be sure you've installed all needed packages
