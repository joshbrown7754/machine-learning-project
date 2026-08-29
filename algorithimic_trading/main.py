import data_collection as dc
import neural_network as neuro
import pyqtgraph as pg

mode = int(input("what mode do you want to run the program in? (1) training or (2) out of sample testing or (3) prediction: "))
new_data = input("do u want to update the dataset? ")
tickers = dc.ticker_request()
if mode == 1:
    if new_data == "yes":
        dc.dataset(tickers)
        dataset_path = r"D:\machine-learning-project\algorithimic_trading\dataset.json"
        dc.restructure_dataset(dataset_path)
        dc.moving_average("dataset.json")
        dc.percentage_change("dataset.json")
    neuro.prediction_neural_net()
elif mode == 2:
    sp500_balance = 10000
    model_balance = 10000
    dc.backtest_dataset(tickers)
    dataset_path = r"D:\algorithimic trading\backtest_data.json"
    dc.restructure_dataset(dataset_path)
    dc.backtest_comparison()
    dc.moving_average("backtest_data.json")
    dc.percentage_change("backtest_data.json")
    app = pg.mkQApp("simple window")

    win = pg.plot(title="sp 500 vs neauralnet prediction")
    win.setWindowTitle("sp 500 vs neauralnet prediction")
    win.setLabel('left', 'Balance', units='$')
    win.setLabel('bottom', 'Days')
    win.plot(sp500_balance,pen = "g")
    win.plot(model_balance,pen = "b")
    
    neural_net_prediction = neuro.prediction_neural_net(training = False,dataset = "backtest_data.json")
    sp500_daily_change = dc.backtest_comparison()
    for i in range (len(neural_net_prediction)):
        model_balance = model_balance * (1+neural_net_prediction[i])
        sp500_balance = sp500_balance * (1+sp500_daily_change[i])
        
        win.plot(sp500_balance,pen = "g")
        win.plot(model_balance,pen = "b")
    
