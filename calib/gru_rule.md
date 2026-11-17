# GRU hybrid: rule and result

Rule, stated in the team's chat at about 02:37 (UTC+4) on 18 November before any GRU number, and committed
here with the result: the GRU hybrid (D from v31; elsewhere the BiGRU's label where it says H or A, otherwise v31's)
replaces the first hybrid only if its held-out spellF1 gain over the first hybrid on the ten public matches is
>= +0.010, positive on >= 8 of 10, positive for Metrica and for DFL, and everything is pushed and checked by 02:56.

Model: bidirectional GRU, 2 layers x 64 units, over 30 s windows (750 frames) of the 67 features of possession/learned.py,
trained on the ten degraded public matches (three seeds, team swap), 800 steps, AdamW; leave-one-match-out for the test
(calib/gru_lomo.py, output calib/gru_lomo_output.txt); final model on all ten (calib/gru_train.py) -> calib/gru_model.npz.

Result: held-out mean spellF1 v31 0.4619, first hybrid 0.4954, GRU alone 0.6559, GRU hybrid 0.5639;
GRU hybrid vs first hybrid +0.0685, 10 of 10; Metrica and DFL both positive. Competition (label-free): 293 possessions
(v31 284), D 47.5%, P(D | alive tags) 8.7%, P(D | set piece) 94.6%, < 2 s 16%, side agreement 80.2%.
numpy inference equals PyTorch within 1.7e-7. Submitted file SHA-256 2824617531b9bcda2e2be323294df14c631589fe21ff40fed279cac825065504.
