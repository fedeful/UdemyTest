import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM
from tensorflow.keras.callbacks import EarlyStopping


def prediction():
    df = pd.read_csv('./RSCCASN.csv', parse_dates=True, index_col='DATE')
    df.columns = ['Sales']
    df = df.iloc[:-30, :]
    print(df.info())
    print(len(df))

    #df.plot()
    #plt.show()

    test_size = 18
    test_ind = len(df) - test_size

    train = df.iloc[:test_ind]
    test = df.iloc[test_ind:]

    # print(test)
    scaler = MinMaxScaler()

    scaler.fit(train)
    scaled_train = scaler.transform(train)
    scaled_test = scaler.transform(test)

    # print(len(test))
    length = 12
    generator = TimeseriesGenerator(scaled_train, scaled_train, length=length, batch_size=1)

    x, y = generator[0]

    # print(x)
    # print(y)
    early_stop = EarlyStopping(monitor='val_loss', patience=2)
    validation_generator = TimeseriesGenerator(scaled_test, scaled_test, length=length, batch_size=1)

    n_features = 1

    model = Sequential()
    model.add(LSTM(100, activation='relu', input_shape=(length, n_features)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    model.summary()

    model.fit_generator(generator=generator, epochs=20, validation_data=validation_generator, callbacks=[early_stop])

    losses = pd.DataFrame(model.history.history)
    # losses.plot()
    # plt.show()

    test_predictions = []
    first_eval_batch = scaled_train[-length:]
    current_batch = first_eval_batch.reshape((1, length, n_features))

    for i in range(len(test)):
        current_pred = model.predict(current_batch)[0]
        test_predictions.append(current_pred)

        current_batch = np.append(current_batch[:, 1:, :], [[current_pred]], axis=1)

    true_prediction = scaler.inverse_transform(test_predictions)

    test['Predictions'] = true_prediction

    test.plot(figsize=(12, 8))
    plt.show()


if __name__ == '__main__':
    prediction()



