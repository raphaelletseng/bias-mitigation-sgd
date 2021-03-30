
"""
#-----------------------------------------------
    def create_model():
        model = RegressionModel(emb_szs=cat_emb_size,
                        n_cont=num_conts,
                        emb_drop=0.04,
                        out_sz=1,
                        szs=[1000, 500, 250],
                        drops=[0.001, 0.01, 0.01],
                        y_range=(0, 1)).to(device)


        class SampleWeightNeuralNet(NeuralNetClassifier):
            def __init__(self, *args, critierion__reduce = False, **kwargs):
                super().__init__(*args, criterion__reduce=criterion__reduce, **kwargs)

            def fit(self, X, y, sample_weight=None):
                if isinstance(X, (pd.DataFrame, pd.Series)):
                    X = X.to_numpy().astype('float32')
                if isinstance(y, (pd.DataFrame, pd.Series)):
                    y = y.to_numpy()
                if sample_weight is not None and isinstance(sample_weight, (pd.DataFrame, pd.Series)):
                    sample_weight = sample_weight.to_numpy()
                y = y.reshape([-1, 1])

                sample_weight = sample_weight if sample_weight is not None else np.ones_like(y)
                X = {'X': X, 'sample_weight': sample_weight}
                return super().fit(X, y)

            def predict(self, X):
                if isinstance(X, (pd.DataFrame, pd.Series)):
                    X = X.to_numpy().astype('float32')
            # The base implementation uses np.argmax which works
            # for multiclass classification only.
                return (super().predict_proba(X) > 0.5).astype(np.float)

            def get_loss(self, y_pred, y_true, X, *args, **kwargs):
            # override get_loss to use the sample_weight from X
                loss_unreduced = super().get_loss(y_pred, y_true.float(), X, *args, **kwargs)
                sample_weight = X['sample_weight']
                sample_weight = sample_weight.to(loss_unreduced.device).unsqueeze(-1)
            # Need to put the sample weights on GPU
                loss_reduced = (sample_weight * loss_unreduced).mean()
                return loss_reduced

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        net = SampleWeightNeuralNet(
            RegressionModel,
            max_epochs=10,
            optimizer=optim.Adam,
            lr=0.001,
            batch_size=512,
            # No validation
            train_split=None,
            # Shuffle training data on each epoch
            iterator_train__shuffle=True,
            criterion=nn.BCELoss,
            device=device
            )
        return net
#-------------------------------------

    def test_gridsearch_classification():
        estimator = create_models()
        disparity_moment = DemographicParity();
    ptc.run_gridsearch_classification(estimator, disparity_moment)

    #Bias mitigation
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = SampleWeightNeuralNet(
        RegressionModel(emb_szs=cat_emb_size,
                        n_cont=num_conts,
                        emb_drop=0.04,
                        out_sz=1,
                        szs=[1000, 500, 250],
                        drops=[0.001, 0.01, 0.01],
                        y_range=(0, 1)),

        max_epochs = 20,
        optimizer = optimizer,
        lr = 0.001,
        iterator_train__shuffle = True,
        criterion = nn.BCELoss,
        device = device
    #    device = device,

        #batch_size = 512,
        #train_split = None,
    )#.to(device)

# X = attributes, y = labels
    #optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = SampleWeightNeuralNet(
        RegressionModel,
        max_epochs=20,
        optimizer = optimizer,
        lr = 0.001,
        iterator_train_shuffle = True,
        criterion = nn.BCELoss,
        device = device
    )

    fit = net.fit(sensitive_idx, y) #y labels) # X, y
    print("\n Fit:", fit)
    y_pred = net.predict(train_data) # y
    print("\n predict:", y_pred)
"""
# FAIRLEARN MITIGATION ---------------------- #
#    np.random.seed(0)
#    constraint = DemographicParity()
#    classifier = net
#    mitigator = ExponentiatedGradient(classifier, constraint)
#    mitigator.fit(train_data, y_true, sensitive_features = sensitive_idx)
#    y_pred_mitigated = mitigator.predict

#    sr_mitigated = MetricFrame()
#    print(sr_mitigated.overall)
#    print(sr_mitigated.by_group)
