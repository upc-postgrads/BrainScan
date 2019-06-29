import train
import argparse


#Training parameters
TRAININGDIR = "/home/deivit/Desktop/dades/Documents/david/upc/AIDL/projecte/unet/BrainTumourImages/Generated_50-50"
LOGDIR = '/tmp/aidl/current'
NUM_EPOCHS = 5 
BATCH_SIZE_TRAIN = 25
BATCH_SIZE_TEST = 25
BATCH_SIZE_VALID = 25
STEP_METRICS = 100
LEARNING_RATE = 1e-5
STEPS_SAVER = 100
MODEL_TO_USE = "unet_tensorflow"
RESTORE_WEIGHTS=False
PERFORM_ONE_HOT=True
BINARIZE_LABELS=False


#The parameters can be introduced by the user
if __name__ == '__main__':    
    parser = argparse.ArgumentParser(description='Pipeline execution')
    parser.add_argument('-t', '--trainingdir', default=TRAININGDIR, help='Location of the TFRecors for training')
    parser.add_argument('-l', '--logdir', default=LOGDIR, help='Log dir for tfevents')
    parser.add_argument('-e', '--num_epochs', type=int, default=NUM_EPOCHS, help='Number of epochs')
    parser.add_argument('-btr', '--size_batch_train', type=int, default=BATCH_SIZE_TRAIN, help='Batch size for training')
    parser.add_argument('-bts', '--size_batch_test', type=int, default=BATCH_SIZE_TEST, help='Batch size for testing')
    parser.add_argument('-bval', '--size_batch_valid', type=int, default=BATCH_SIZE_VALID, help='Batch size for validation')
    parser.add_argument('-sm', '--step_metrics', type=int, default=STEP_METRICS, help='frequency of summary within training batches')
    parser.add_argument('-ss', '--steps_saver', type=int, default=STEPS_SAVER, help='frequency of weight saving within training batches')
    parser.add_argument('-lr', '--learning_rate', type=float, default=LEARNING_RATE, help='Learning rate')
    parser.add_argument('-r', '--restore_weights', type=float, default=RESTORE_WEIGHTS, help='Restore weights from logdir path.')
    parser.add_argument('-m', '--model', default=MODEL_TO_USE,help='Model to use, either unet_keras or unet_tensorflow.')
    parser.add_argument('-oh', '--perform_one_hot', default=PERFORM_ONE_HOT,help='Perform on-hot encoding for labels.')
    parser.add_argument('-bi', '--binarize_labels', default=BINARIZE_LABELS,help='Perform binarization for labels.')
    
    args = parser.parse_args()

    print("Are we testing or training? True for training, False for testing")
    
    training = raw_input('Enter your input:') 
    if training :
    
        #we call the train function
        train.main(args.trainingdir,args.model,args.num_epochs,args.size_batch_train,args.size_batch_test,args.size_batch_valid, args.step_metrics, args.steps_saver,args.learning_rate, args.logdir, args.restore_weights,args.perform_one_hot,args.binarize_labels)
    
    else:
    
        #we call the test function    
        test.main(args.trainingdir,args.model,args.num_epochs,args.size_batch_train,args.size_batch_test,args.size_batch_valid, args.logdir,args.perform_one_hot,args.binarize_labels)

