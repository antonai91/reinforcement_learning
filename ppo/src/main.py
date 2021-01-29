
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        # Save the model, I need this in order to save the networks, frame number, rewards and losses. 
        # if I want to stop the script and restart without training from the beginning
        if PATH_SAVE_MODEL is None:
            print("Setting path to ../model")
            PATH_SAVE_MODEL = "../model"
        print('Saving the model in ' + f'{PATH_SAVE_MODEL}/save_agent_{time.strftime("%Y%m%d%H%M")}')
        ppo_agent.save_model(f'{PATH_SAVE_MODEL}/save_agent_{time.strftime("%Y%m%d%H%M")}')
        print('Saved.')
