from training.reward import get_reward
from training.agent_func import load_static_memory, MEMORY_PATH

def main():
    question = "Who is Lisa Schmidt to my niece, Sophie de Vries?"
    agent_reply = "Lisa Schmidt is a friend of Sophie de Vries."
    ground_truth = "friend(Sophie de Vries)"
    reward = get_reward(question, agent_reply, ground_truth, debug=True)
    print(reward)

    static_memory = load_static_memory()
    static_memory.reset(MEMORY_PATH)

if __name__ == "__main__":
    main()
