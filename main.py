from create_data import *
import requests
import json
from sklearn.model_selection import train_test_split

dfeco = read_data_file()

DEVICE_ID = 296
tmp_df = dfeco.query('device_id == @DEVICE_ID').sort_values('ts')

X = tmp_df.drop(non_model_drops, axis=1)
y = tmp_df['ind_shutdown_mal_3D_next']

dfeco_train, dfeco_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


payload = dfeco_test.iloc[:1, :].to_dict()
response = requests.get(
    'http://127.0.0.1:8080/predict',
    json=payload
)

print(response.status_code)
print(json.loads(response.text))


dfeco_test['label'] = y_test
payload = dfeco_test.to_dict()


response = requests.post(
    'http://127.0.0.1:8080/train_model',
    json=payload
)






