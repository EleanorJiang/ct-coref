import pickle, os
from util.util import whether_in_the_list
import argparse
import jsonlines, json
from get_prediction import load_predictor_from_th, get_prediction_from_model

def train_by_allennlp(config_path, archive_path):
    c = f"python -m allennlp.run train {config_path} -s {archive_path}"
    os.system(c)


def move_th_out_of_archieve(archive_path):
    if(os.path.isdir(os.path.join(archive_path, "archieve"))):
        for file in os.listdir(os.path.join(archive_path, "archieve")):
            if not os.path.isfile(os.path.join(archive_path, file)):
                os.rename(os.path.join(archive_path, "archieve", file),
                          os.path.join(archive_path, file))


def evaluated_by_allennlp(archive_path, data_dir, only_best=False):
    cuda_device = 0
    test_path = os.path.join(data_dir, "test.english.v4_gold_conll")
    for file in os.listdir(archive_path):
        if not file.endswith(".th") or not (file.startswith("model_state_epoch_") or file.startswith("best")):
            continue
        if only_best and (not file.startswith("best")):
            continue
        weights_name = file[:-3]
        weight_file = os.path.join(archive_path, file)
        output_file = os.path.join(archive_path, "{}_test_metric.json".format(weights_name))
        predictions_output_file = os.path.join(archive_path, "{}_predictions.jsonl".format(weights_name))
        if(os.path.isfile(output_file)):
            print(output_file, " exists")
            continue
        print("Evaluating ", weight_file)
        c = f"allennlp evaluate {archive_path} {test_path}  --weights-file \"{weight_file}\" --output-file \"{output_file}\" --predictions-output-file \"{predictions_output_file}\" --cuda-device {cuda_device}"
        os.system(c)


def evaluated_by_allennlp_per_documents(archive_path, data_dir, only_best=True, include_list=None):
    cuda_device = 0
    for file in os.listdir(archive_path):
        if not file.endswith(".th") or not (file.startswith("model_state_epoch_") or file.startswith("best")):
            continue
        if only_best and (not file.startswith("best")):
            continue
        if whether_in_the_list(file, include_list) is False:
            continue
        weights_name = file[:-3]
        weight_file = os.path.join(archive_path, file)
        for i in range(348):
            test_path = os.path.join(data_dir, "test.english.v4_gold_conll_{}".format(i)) # "/cluster/work/cotterell/ct/data/ontonotes/test.english.v4_gold_conll_0"
            output_file = os.path.join(archive_path, "{}_test_metric_{}.json".format(weights_name, i))
            predictions_output_file = os.path.join(archive_path, "{}_predictions_{}.jsonl".format(weights_name,i))
            if(os.path.isfile(output_file)):
                print(output_file, " exists")
                continue
            print("Evaluating ", weight_file, i)
            c = f"allennlp evaluate {archive_path} {test_path}  --weights-file \"{weight_file}\" --output-file \"{output_file}\" --predictions-output-file \"{predictions_output_file}\" --cuda-device {cuda_device}"
            os.system(c)


def get_predicted_dict(archive_path, tmp_dir, override = False):
    print("getting predicted dict...")
    with open(os.path.join(tmp_dir, 'documents.data'), 'rb') as filehandle:
        documents = pickle.load(filehandle)
    # Predict:
    for file in os.listdir(archive_path):
        if not file.endswith(".th") or not (file.startswith("model_state_epoch_") or file.startswith("best")):
            continue
        # start
        weights_name = file[:-3]
        weights_file = os.path.join(archive_path, file)
        output_file = os.path.join(
            archive_path, "{}_predicted_dicts.data".format(weights_name))
        if os.path.isfile(output_file) and override is False:
            print(output_file, " exists")
            continue
        predictor = load_predictor_from_th(
            archive_path, weights_file, cuda_device=0)
        predicted_dicts = get_prediction_from_model(
            documents, predictor)
        print(os.path.join(archive_path, weights_file), len(predicted_dicts))
        with open(output_file, 'wb') as filehandle:
            pickle.dump(predicted_dicts, filehandle)


def get_predicted_dict_from_jsonl(archive_path,override=True):
    print("getting predicted dict...")
    # Predict:
    for file in os.listdir(archive_path):
        if not file.endswith(".th") or not (file.startswith("model_state_epoch_") or file.startswith("best")):
            continue
        # start
        weights_name = file[:-3]
        jsonl_file = os.path.join(archive_path, f"{weights_name}_predictions.jsonl")
        output_file = os.path.join(archive_path, f"{weights_name}_predicted_dicts.data")
        if os.path.isfile(output_file) and override is False:
            print(output_file, " exists")
            continue
        predicted_dicts = []
        with jsonlines.open(jsonl_file) as products:
            for prod in products:
                jdump = json.dumps(prod)
                jload = json.loads(jdump)
                predicted_dicts.append(jload)
        print(os.path.join(archive_path, jsonl_file), len(predicted_dicts))
        with open(output_file, 'wb') as filehandle:
            pickle.dump(predicted_dicts, filehandle)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tmp_dir", help="the path of saved tmp results", type=str, default="/cluster/work/cotterell/ct/data/centering_exp"
    )
    parser.add_argument("--data_dir", help="the path of the ontonotes dataset", type=str, default="/cluster/work/cotterell/ct/data/ontonotes")
    parser.add_argument(
        "--exp_type", type=str, default="gold-mention-2021.8.2", help=" exp_type + exp_date"
    )
    parser.add_argument(
        "--ids", type=str, default="100", help="the percentage of training data, e.g. 10,20,30"
    )
    parser.add_argument(
        "--ths", type=str, default="0", help="the th checkpoints we want to exam, e.g. 0,1,2,best",
    )
    args = parser.parse_args()
    id_list = [item for item in args.ids.split(',')]
    archive_paths = [os.path.join(args.data_dir, f"{args.exp_type}.{idx}") for idx in id_list]
    if args.ths != "":
        include_list = [item for item in args.ths.split(',')]
    else:
        include_list = None

    for archive_path in archive_paths:
        # move_th_out_of_archieve(archive_path)
        # evaluated_by_allennlp(archive_path, args.data_dir, only_best= False)
        # evaluated_by_allennlp_per_documents(archive_path, only_best=True)
        get_predicted_dict(archive_path, args.tmp_dir, override=True)
        # get_predicted_dict_from_jsonl(archive_path)
        # being a bit lazy：directly run c2f_analysis below
        # c2f_analysis.c2f_analysis(archive_path)
