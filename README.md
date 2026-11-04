# AlphaGo Zero Lite

## Enviroment
To avoid version control issues, use the following commands to load the python enviroment for this project:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
If you add pakages please run before committing:
```bash
pip freeze > requirements.txt
```

## Running Code

To train, please use the following command:

```python -m train --game tic_tac_toe```

To play, please use the following command:

```python -m play --game tic_tac_toe```



