import autoencoder as ae

data = ae.preprocess()
test = ae.load_test_batch()
auto_encoder = ae.Model(32, 32, 3)
auto_encoder.cuda(ae.device)
ae.training(auto_encoder, data, test)