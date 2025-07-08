from training.reward import get_reward

def main():
    agent_reply = "Lisa Schmidt is a friend of Sophie de Vries."
    ground_truth = "friend(Sophie de Vries)"
    reward = get_reward(agent_reply, ground_truth, debug=True)
    print(reward)

if __name__ == "__main__":
    main()
