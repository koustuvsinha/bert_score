from bert_score import score, score_with_two_models

if __name__ == '__main__':
    hyps = []
    with open('example/hyps.txt') as fp:
        for line in fp:
            hyps.append(line.rstrip())
    refs = []
    with open('example/refs.txt') as fp:
        for line in fp:
            refs.append(line.rstrip())

    # preds = score(hyps,
    #     refs,
    #     model_type='roberta-base',
    #     verbose=True,
    #     idf=True,
    #     device='cpu',
    #     batch_size=4,
    #     nthreads=4,
    #     return_hash=False,
    #     rescale_with_baseline=False,
    #     baseline_path=None)

#     tensor([0.9734, 0.9789, 0.9176, 0.9520, 0.9410, 0.9666, 0.9392, 0.9650, 0.9572,
#         0.9464])
# 1:tensor([0.9740, 0.9634, 0.9301, 0.9557, 0.9673, 0.9694, 0.9746, 0.9650, 0.9584,
#         0.9532])
# 2:tensor([0.9737, 0.9711, 0.9238, 0.9538, 0.9540, 0.9680, 0.9566, 0.9650, 0.9578,

# 0:tensor([0.9734, 0.9789, 0.9176, 0.9520, 0.9410, 0.9666, 0.9392, 0.9650, 0.9572,
#         0.9464])
# 1:tensor([0.9740, 0.9634, 0.9301, 0.9557, 0.9673, 0.9694, 0.9746, 0.9650, 0.9584,
#         0.9532])
# 2:tensor([0.9737, 0.9711, 0.9238, 0.9538, 0.9540, 0.9680, 0.9566, 0.9650, 0.9578,

    preds = score_with_two_models(hyps,
        refs,
        model_type_1='roberta-base',
        model_type_2='roberta-base',
        verbose=True,
        idf=True,
        device='cpu',
        batch_size=4,
        nthreads=4,
        return_hash=False,
        rescale_with_baseline=False,
        baseline_path=None)

    print("Done")