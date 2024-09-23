import json
import matplotlib.pyplot as plt

# Sample JSON data
data = '''{
  "best_metric": null,
  "best_model_checkpoint": null,
  "epoch": 6.0,
  "eval_steps": 500,
  "global_step": 43194,
  "is_hyper_param_search": false,
  "is_local_process_zero": true,
  "is_world_process_zero": true,
  "log_history": [
    {
      "epoch": 0.06945409084595083,
      "grad_norm": 2.7140793800354004,
      "learning_rate": 1.976848636384683e-05,
      "loss": 3.2723,
      "step": 500
    },
    {
      "epoch": 0.13890818169190167,
      "grad_norm": 5.291518211364746,
      "learning_rate": 1.9536972727693665e-05,
      "loss": 3.15,
      "step": 1000
    },
    {
      "epoch": 0.20836227253785247,
      "grad_norm": 3.180028200149536,
      "learning_rate": 1.9305459091540492e-05,
      "loss": 3.0586,
      "step": 1500
    },
    {
      "epoch": 0.27781636338380333,
      "grad_norm": 4.574570655822754,
      "learning_rate": 1.9073945455387325e-05,
      "loss": 2.978,
      "step": 2000
    },
    {
      "epoch": 0.3472704542297541,
      "grad_norm": 6.445160388946533,
      "learning_rate": 1.8842431819234155e-05,
      "loss": 2.9318,
      "step": 2500
    },
    {
      "epoch": 0.41672454507570494,
      "grad_norm": 7.676687240600586,
      "learning_rate": 1.8610918183080985e-05,
      "loss": 2.8749,
      "step": 3000
    },
    {
      "epoch": 0.4861786359216558,
      "grad_norm": 8.3671293258667,
      "learning_rate": 1.8379404546927815e-05,
      "loss": 2.7855,
      "step": 3500
    },
    {
      "epoch": 0.5556327267676067,
      "grad_norm": 32.55449295043945,
      "learning_rate": 1.8147890910774645e-05,
      "loss": 2.7129,
      "step": 4000
    },
    {
      "epoch": 0.6250868176135574,
      "grad_norm": 5.203619003295898,
      "learning_rate": 1.791637727462148e-05,
      "loss": 2.6265,
      "step": 4500
    },
    {
      "epoch": 0.6945409084595082,
      "grad_norm": 27.798006057739258,
      "learning_rate": 1.768486363846831e-05,
      "loss": 2.5648,
      "step": 5000
    },
    {
      "epoch": 0.7639949993054591,
      "grad_norm": 64.46216583251953,
      "learning_rate": 1.7453350002315135e-05,
      "loss": 2.4637,
      "step": 5500
    },
    {
      "epoch": 0.8334490901514099,
      "grad_norm": 24.43657875061035,
      "learning_rate": 1.722183636616197e-05,
      "loss": 2.3691,
      "step": 6000
    },
    {
      "epoch": 0.9029031809973608,
      "grad_norm": 70.89596557617188,
      "learning_rate": 1.69903227300088e-05,
      "loss": 2.2392,
      "step": 6500
    },
    {
      "epoch": 0.9723572718433116,
      "grad_norm": 92.28575134277344,
      "learning_rate": 1.675880909385563e-05,
      "loss": 2.1275,
      "step": 7000
    },
    {
      "epoch": 1.0,
      "eval_loss": 3.384153127670288,
      "eval_runtime": 4910.4302,
      "eval_samples_per_second": 5.212,
      "eval_steps_per_second": 0.163,
      "step": 7199
    },
    {
      "epoch": 1.0418113626892624,
      "grad_norm": 58.799678802490234,
      "learning_rate": 1.652729545770246e-05,
      "loss": 2.0229,
      "step": 7500
    },
    {
      "epoch": 1.1112654535352133,
      "grad_norm": 81.19511413574219,
      "learning_rate": 1.629578182154929e-05,
      "loss": 1.9043,
      "step": 8000
    },
    {
      "epoch": 1.180719544381164,
      "grad_norm": 94.4384994506836,
      "learning_rate": 1.6064268185396122e-05,
      "loss": 1.8247,
      "step": 8500
    },
    {
      "epoch": 1.2501736352271149,
      "grad_norm": 79.72384643554688,
      "learning_rate": 1.5832754549242952e-05,
      "loss": 1.7434,
      "step": 9000
    },
    {
      "epoch": 1.3196277260730658,
      "grad_norm": 28.40268325805664,
      "learning_rate": 1.5601240913089782e-05,
      "loss": 1.6719,
      "step": 9500
    },
    {
      "epoch": 1.3890818169190164,
      "grad_norm": 17.943021774291992,
      "learning_rate": 1.5369727276936612e-05,
      "loss": 1.6386,
      "step": 10000
    },
    {
      "epoch": 1.4585359077649673,
      "grad_norm": 58.01112365722656,
      "learning_rate": 1.5138213640783444e-05,
      "loss": 1.559,
      "step": 10500
    },
    {
      "epoch": 1.5279899986109182,
      "grad_norm": 28.444774627685547,
      "learning_rate": 1.4906700004630274e-05,
      "loss": 1.4975,
      "step": 11000
    },
    {
      "epoch": 1.5974440894568689,
      "grad_norm": 39.57249069213867,
      "learning_rate": 1.4675186368477104e-05,
      "loss": 1.4671,
      "step": 11500
    },
    {
      "epoch": 1.66689818030282,
      "grad_norm": 44.92505645751953,
      "learning_rate": 1.4443672732323934e-05,
      "loss": 1.4046,
      "step": 12000
    },
    {
      "epoch": 1.7363522711487707,
      "grad_norm": 35.30231475830078,
      "learning_rate": 1.4212159096170766e-05,
      "loss": 1.3717,
      "step": 12500
    },
    {
      "epoch": 1.8058063619947213,
      "grad_norm": 91.458740234375,
      "learning_rate": 1.3980645460017596e-05,
      "loss": 1.3442,
      "step": 13000
    },
    {
      "epoch": 1.8752604528406724,
      "grad_norm": 82.4042739868164,
      "learning_rate": 1.3749131823864428e-05,
      "loss": 1.3213,
      "step": 13500
    },
    {
      "epoch": 1.944714543686623,
      "grad_norm": 43.93354797363281,
      "learning_rate": 1.3517618187711256e-05,
      "loss": 1.2825,
      "step": 14000
    },
    {
      "epoch": 2.0,
      "eval_loss": 2.931705951690674,
      "eval_runtime": 8437.801,
      "eval_samples_per_second": 3.033,
      "eval_steps_per_second": 0.095,
      "step": 14398
    },
    {
      "epoch": 2.0141686345325738,
      "grad_norm": 15.423131942749023,
      "learning_rate": 1.3286104551558088e-05,
      "loss": 1.2666,
      "step": 14500
    },
    {
      "epoch": 2.083622725378525,
      "grad_norm": 43.316505432128906,
      "learning_rate": 1.305459091540492e-05,
      "loss": 1.2442,
      "step": 15000
    },
    {
      "epoch": 2.1530768162244756,
      "grad_norm": 14.179651260375977,
      "learning_rate": 1.282307727925175e-05,
      "loss": 1.2208,
      "step": 15500
    },
    {
      "epoch": 2.2225309070704267,
      "grad_norm": 15.007329940795898,
      "learning_rate": 1.259156364309858e-05,
      "loss": 1.1931,
      "step": 16000
    },
    {
      "epoch": 2.2919849979163773,
      "grad_norm": 72.26441955566406,
      "learning_rate": 1.236005000694541e-05,
      "loss": 1.1881,
      "step": 16500
    },
    {
      "epoch": 2.361439088762328,
      "grad_norm": 153.21990966796875,
      "learning_rate": 1.2128536370792241e-05,
      "loss": 1.1476,
      "step": 17000
    },
    {
      "epoch": 2.430893179608279,
      "grad_norm": 74.37097930908203,
      "learning_rate": 1.1897022734639071e-05,
      "loss": 1.1393,
      "step": 17500
    },
    {
      "epoch": 2.5003472704542298,
      "grad_norm": 26.178112030029297,
      "learning_rate": 1.1665509098485901e-05,
      "loss": 1.1185,
      "step": 18000
    },
    {
      "epoch": 2.5698013613001804,
      "grad_norm": 11.019347190856934,
      "learning_rate": 1.1433995462332731e-05,
      "loss": 1.1202,
      "step": 18500
    },
    {
      "epoch": 2.6392554521461316,
      "grad_norm": 25.1142635345459,
      "learning_rate": 1.1202481826179563e-05,
      "loss": 1.0999,
      "step": 19000
    },
    {
      "epoch": 2.708709542992082,
      "grad_norm": 15.312582969665527,
      "learning_rate": 1.0970968190026395e-05,
      "loss": 1.0917,
      "step": 19500
    },
    {
      "epoch": 2.778163633838033,
      "grad_norm": 58.614688873291016,
      "learning_rate": 1.0739454553873223e-05,
      "loss": 1.0731,
      "step": 20000
    },
    {
      "epoch": 2.847617724683984,
      "grad_norm": 74.21333312988281,
      "learning_rate": 1.0507940917720055e-05,
      "loss": 1.0674,
      "step": 20500
    },
    {
      "epoch": 2.9170718155299347,
      "grad_norm": 58.53996658325195,
      "learning_rate": 1.0276427281566885e-05,
      "loss": 1.0472,
      "step": 21000
    },
    {
      "epoch": 2.9865259063758858,
      "grad_norm": 36.44721221923828,
      "learning_rate": 1.0044913645413717e-05,
      "loss": 1.0267,
      "step": 21500
    },
    {
      "epoch": 3.0,
      "eval_loss": 3.015873908996582,
      "eval_runtime": 5122.4627,
      "eval_samples_per_second": 4.997,
      "eval_steps_per_second": 0.156,
      "step": 21597
    },
    {
      "epoch": 3.0559799972218364,
      "grad_norm": 15.47930908203125,
      "learning_rate": 9.813400009260547e-06,
      "loss": 1.0211,
      "step": 22000
    },
    {
      "epoch": 3.125434088067787,
      "grad_norm": 21.26850700378418,
      "learning_rate": 9.581886373107377e-06,
      "loss": 1.0108,
      "step": 22500
    },
    {
      "epoch": 3.194888178913738,
      "grad_norm": 31.73493766784668,
      "learning_rate": 9.350372736954207e-06,
      "loss": 1.0096,
      "step": 23000
    },
    {
      "epoch": 3.264342269759689,
      "grad_norm": 51.030609130859375,
      "learning_rate": 9.118859100801037e-06,
      "loss": 0.9892,
      "step": 23500
    },
    {
      "epoch": 3.3337963606056396,
      "grad_norm": 16.116079330444336,
      "learning_rate": 8.887345464647869e-06,
      "loss": 0.9911,
      "step": 24000
    },
    {
      "epoch": 3.4032504514515907,
      "grad_norm": 10.373322486877441,
      "learning_rate": 8.655831828494699e-06,
      "loss": 0.9785,
      "step": 24500
    },
    {
      "epoch": 3.4727045422975413,
      "grad_norm": 21.92342758178711,
      "learning_rate": 8.42431819234153e-06,
      "loss": 0.9735,
      "step": 25000
    },
    {
      "epoch": 3.542158633143492,
      "grad_norm": 56.352210998535156,
      "learning_rate": 8.19280455618836e-06,
      "loss": 0.9627,
      "step": 25500
    },
    {
      "epoch": 3.611612723989443,
      "grad_norm": 50.64139938354492,
      "learning_rate": 7.96129092003519e-06,
      "loss": 0.9612,
      "step": 26000
    },
    {
      "epoch": 3.6810668148353938,
      "grad_norm": 164.03485107421875,
      "learning_rate": 7.729777283882022e-06,
      "loss": 0.9454,
      "step": 26500
    },
    {
      "epoch": 3.750520905681345,
      "grad_norm": 27.826683044433594,
      "learning_rate": 7.498263647728851e-06,
      "loss": 0.9329,
      "step": 27000
    },
    {
      "epoch": 3.8199749965272956,
      "grad_norm": 27.45733642578125,
      "learning_rate": 7.266750011575683e-06,
      "loss": 0.9307,
      "step": 27500
    },
    {
      "epoch": 3.889429087373246,
      "grad_norm": 19.011194229125977,
      "learning_rate": 7.035236375422513e-06,
      "loss": 0.9209,
      "step": 28000
    },
    {
      "epoch": 3.958883178219197,
      "grad_norm": 35.23545837402344,
      "learning_rate": 6.803722739269344e-06,
      "loss": 0.9236,
      "step": 28500
    },
    {
      "epoch": 4.0,
      "eval_loss": 2.7699551582336426,
      "eval_runtime": 4921.0695,
      "eval_samples_per_second": 5.201,
      "eval_steps_per_second": 0.163,
      "step": 28796
    },
    {
      "epoch": 4.0283372690651476,
      "grad_norm": 50.9335823059082,
      "learning_rate": 6.572209103116174e-06,
      "loss": 0.9212,
      "step": 29000
    },
    {
      "epoch": 4.097791359911099,
      "grad_norm": 8.253153800964355,
      "learning_rate": 6.340695466963005e-06,
      "loss": 0.8937,
      "step": 29500
    },
    {
      "epoch": 4.16724545075705,
      "grad_norm": 20.836532592773438,
      "learning_rate": 6.109181830809835e-06,
      "loss": 0.8872,
      "step": 30000
    },
    {
      "epoch": 4.236699541603,
      "grad_norm": 9.500503540039062,
      "learning_rate": 5.877668194656666e-06,
      "loss": 0.8926,
      "step": 30500
    },
    {
      "epoch": 4.306153632448951,
      "grad_norm": 24.964262008666992,
      "learning_rate": 5.646154558503496e-06,
      "loss": 0.89,
      "step": 31000
    },
    {
      "epoch": 4.375607723294902,
      "grad_norm": 51.99266052246094,
      "learning_rate": 5.4146409223503275e-06,
      "loss": 0.8782,
      "step": 31500
    },
    {
      "epoch": 4.445061814140853,
      "grad_norm": 11.249212265014648,
      "learning_rate": 5.183127286197157e-06,
      "loss": 0.8757,
      "step": 32000
    },
    {
      "epoch": 4.514515904986804,
      "grad_norm": 6.954973220825195,
      "learning_rate": 4.9516136500439885e-06,
      "loss": 0.8681,
      "step": 32500
    },
    {
      "epoch": 4.583969995832755,
      "grad_norm": 16.269723892211914,
      "learning_rate": 4.7201000138908185e-06,
      "loss": 0.8728,
      "step": 33000
    },
    {
      "epoch": 4.653424086678705,
      "grad_norm": 12.936798095703125,
      "learning_rate": 4.488586377737649e-06,
      "loss": 0.8627,
      "step": 33500
    },
    {
      "epoch": 4.722878177524656,
      "grad_norm": 28.665285110473633,
      "learning_rate": 4.257072741584479e-06,
      "loss": 0.8662,
      "step": 34000
    },
    {
      "epoch": 4.792332268370607,
      "grad_norm": 28.082307815551758,
      "learning_rate": 4.02555910543131e-06,
      "loss": 0.8577,
      "step": 34500
    },
    {
      "epoch": 4.861786359216558,
      "grad_norm": 19.575767517089844,
      "learning_rate": 3.7940454692781407e-06,
      "loss": 0.8502,
      "step": 35000
    },
    {
      "epoch": 4.931240450062509,
      "grad_norm": 30.053512573242188,
      "learning_rate": 3.5625318331249716e-06,
      "loss": 0.8497,
      "step": 35500
    },
    {
      "epoch": 5.0,
      "eval_loss": 2.8077614307403564,
      "eval_runtime": 8237.6651,
      "eval_samples_per_second": 3.107,
      "eval_steps_per_second": 0.097,
      "step": 35995
    },
    {
      "epoch": 5.0006945409084596,
      "grad_norm": 37.28679656982422,
      "learning_rate": 3.331018196971802e-06,
      "loss": 0.842,
      "step": 36000
    },
    {
      "epoch": 5.07014863175441,
      "grad_norm": 8.623671531677246,
      "learning_rate": 3.0995045608186325e-06,
      "loss": 0.8362,
      "step": 36500
    },
    {
      "epoch": 5.139602722600361,
      "grad_norm": 42.12864685058594,
      "learning_rate": 2.867990924665463e-06,
      "loss": 0.8252,
      "step": 37000
    },
    {
      "epoch": 5.2090568134463116,
      "grad_norm": 15.479891777038574,
      "learning_rate": 2.6364772885122934e-06,
      "loss": 0.8303,
      "step": 37500
    },
    {
      "epoch": 5.278510904292263,
      "grad_norm": 18.80664825439453,
      "learning_rate": 2.4049636523591243e-06,
      "loss": 0.8266,
      "step": 38000
    },
    {
      "epoch": 5.347964995138214,
      "grad_norm": 16.035415649414062,
      "learning_rate": 2.1734500162059548e-06,
      "loss": 0.8259,
      "step": 38500
    },
    {
      "epoch": 5.417419085984164,
      "grad_norm": 25.78068733215332,
      "learning_rate": 1.9419363800527852e-06,
      "loss": 0.8191,
      "step": 39000
    },
    {
      "epoch": 5.486873176830115,
      "grad_norm": 12.028929710388184,
      "learning_rate": 1.7104227438996157e-06,
      "loss": 0.815,
      "step": 39500
    },
    {
      "epoch": 5.556327267676066,
      "grad_norm": 11.543804168701172,
      "learning_rate": 1.4789091077464463e-06,
      "loss": 0.8206,
      "step": 40000
    },
    {
      "epoch": 5.625781358522017,
      "grad_norm": 6.55134391784668,
      "learning_rate": 1.247395471593277e-06,
      "loss": 0.8145,
      "step": 40500
    },
    {
      "epoch": 5.695235449367968,
      "grad_norm": 22.499408721923828,
      "learning_rate": 1.0158818354401075e-06,
      "loss": 0.8115,
      "step": 41000
    },
    {
      "epoch": 5.764689540213919,
      "grad_norm": 13.747957229614258,
      "learning_rate": 7.84368199286938e-07,
      "loss": 0.8188,
      "step": 41500
    },
    {
      "epoch": 5.834143631059869,
      "grad_norm": 16.531089782714844,
      "learning_rate": 5.528545631337686e-07,
      "loss": 0.8063,
      "step": 42000
    },
    {
      "epoch": 5.90359772190582,
      "grad_norm": 14.789374351501465,
      "learning_rate": 3.213409269805992e-07,
      "loss": 0.808,
      "step": 42500
    },
    {
      "epoch": 5.9730518127517715,
      "grad_norm": 5.987864971160889,
      "learning_rate": 8.982729082742975e-08,
      "loss": 0.8124,
      "step": 43000
    },
    {
      "epoch": 6.0,
      "eval_loss": 2.7330167293548584,
      "eval_runtime": 4935.5945,
      "eval_samples_per_second": 5.186,
      "eval_steps_per_second": 0.162,
      "step": 43194
    }
  ],
  "logging_steps": 500,
  "max_steps": 43194,
  "num_input_tokens_seen": 0,
  "num_train_epochs": 6,
  "save_steps": 500,
  "stateful_callbacks": {
    "TrainerControl": {
      "args": {
        "should_epoch_stop": false,
        "should_evaluate": false,
        "should_log": false,
        "should_save": true,
        "should_training_stop": true
      },
      "attributes": {}
    }
  },
  "total_flos": 0.0,
  "train_batch_size": 32,
  "trial_name": null,
  "trial_params": null
}
'''

# Load JSON data
json_data = json.loads(data)

# Extract relevant fields for plotting
epochs = []
losses = []
eval_losses = []
grad_norms = []
learning_rates = []

for entry in json_data['log_history']:
    if 'epoch' in entry:
        epochs.append(entry['epoch'])
    if 'loss' in entry:
        losses.append(entry['loss'])
    else:
        losses.append(None)
    if 'eval_loss' in entry:
        eval_losses.append(entry['eval_loss'])
    else:
        eval_losses.append(None)
    if 'grad_norm' in entry:
        grad_norms.append(entry['grad_norm'])
    else:
        grad_norms.append(None)
    if 'learning_rate' in entry:
        learning_rates.append(entry['learning_rate'])
    else:
        learning_rates.append(None)

# Filter out None values for straight lines
valid_epochs = [epoch for epoch, eval_loss in zip(epochs, eval_losses) if eval_loss is not None]
valid_eval_losses = [eval_loss for eval_loss in eval_losses if eval_loss is not None]

# Plot 1: Loss and Eval Loss vs Epochs with straight line for eval loss
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(epochs, losses, label='Training Loss', marker='o')
plt.plot(valid_epochs, valid_eval_losses, label='Eval Loss', marker='x', linestyle='-')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss and Eval Loss vs Epochs')
plt.legend()

# Plot 2: Grad Norm and Learning Rate vs Epochs
plt.subplot(1, 2, 2)
plt.plot(epochs, grad_norms, label='Grad Norm', marker='o')
plt.plot(epochs, learning_rates, label='Learning Rate', marker='x')
plt.xlabel('Epochs')
plt.ylabel('Grad Norm / Learning Rate')
plt.title('Grad Norm and Learning Rate vs Epochs')
plt.legend()

# Display plots
plt.tight_layout()
plt.show()
