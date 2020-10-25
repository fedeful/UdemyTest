import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM
from tensorflow.keras.callbacks import EarlyStopping


def predict_sine_wave(plot=False, early_stop_boolean=False):
    x = np.linspace(0, 50, 501)
    y = np.sin(x)

    if plot:
        plt.plot(x, y)
        plt.show()

    df = pd.DataFrame(data=y, index=x, columns=['Sine'])
    print(len(df))

    # percentuale utilizzata per i test
    test_percent = 0.1
    print(f"{len(df)*test_percent}")

    test_point = np.round(len(df)*test_percent)
    test_index = int(len(df) - test_point)

    train = df.iloc[:test_index, :]
    test = df.iloc[test_index:, :]

    scaler = MinMaxScaler()

    scaler.fit(train)
    scaled_train = scaler.transform(train)
    scaled_test = scaler.transform(test)

    length = 49
    batch_size = 1

    if early_stop_boolean:
        early_stop = EarlyStopping(monitor='val_loss', patience=2)

        generator = TimeseriesGenerator(scaled_train, scaled_train,
                                        length=length, batch_size=batch_size)

        validation_generator = TimeseriesGenerator(scaled_test, scaled_test,
                                                    length=length, batch_size=batch_size)

    else:
        generator = TimeseriesGenerator(scaled_train, scaled_train,
                                        length=length, batch_size=batch_size)

    print(f"lunghezza train set scalato: {len(scaled_train)}")
    print(f"lunghezza generatore: {len(generator)}")

    x, y = generator[0]

    print(f"x: {x}")
    print(f"y: {y}")

    n_features = 1
    model = Sequential()

    # model.add(SimpleRNN(50, input_shape=(length, n_features)))
    model.add(LSTM(50, input_shape=(length, n_features)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.summary()

    if early_stop_boolean:
        model.fit_generator(generator, validation_data=validation_generator,  callbacks=[early_stop], epochs=20)
    else:
        model.fit_generator(generator, epochs=5)

    losses = pd.DataFrame(model.history.history)
    #losses.plot()
    #plt.show()

    first_eval_batch = scaled_train[-length:]
    first_eval_batch = first_eval_batch.reshape((1, length, n_features))
    # print(f"first prediction: {model.predict(first_eval_batch)}")
    # print(f"firs test: {scaled_test[0]}")

    test_predictions = []
    current_batch = first_eval_batch

    # predicted_value = [[[99]]]
    # np.append(current_batch[:, 1:, :], [[[99]]], axis=1)

    for i in range(len(test)):

        current_pred = model.predict(current_batch)[0]
        test_predictions.append(current_pred)

        current_batch = np.append(current_batch[:, 1:, :], [[current_pred]], axis=1)

    true_predictions = scaler.inverse_transform(test_predictions)
    test['Predicitons'] = true_predictions
    test.plot(figsize=(12, 8))
    plt.show()

    return 0


if __name__ == '__main__':
    predict_sine_wave(early_stop_boolean=True)
