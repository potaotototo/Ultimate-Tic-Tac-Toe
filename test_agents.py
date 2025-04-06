import time
import numpy as np
from utils import State, Action
from fourth_agent import FourthAgent  
from fifth_agent import FifthAgent  
from sixth_agent import SixthAgent
from seventh_agent import SeventhAgent
from eighth_agent import EighthAgent
from nineth_agent import NinethAgent
from tenth_agent import TenthAgent

def run(your_agent, opponent_agent, start_num: int):
    your_agent_stats = {"timeout_count": 0, "invalid_count": 0}
    opponent_agent_stats = {"timeout_count": 0, "invalid_count": 0}
    turn_count = 0
    
    state = State(fill_num=start_num)
    
    while not state.is_terminal():
        turn_count += 1

        agent_name = "your_agent" if state.fill_num == 1 else "opponent_agent"
        agent = your_agent if state.fill_num == 1 else opponent_agent
        stats = your_agent_stats if state.fill_num == 1 else opponent_agent_stats

        start_time = time.time()
        action = agent.choose_action(state.clone())
        end_time = time.time()
        
        random_action = state.get_random_valid_action()
        if end_time - start_time > 3:
            print(f"{agent_name} timed out!")
            stats["timeout_count"] += 1
            action = random_action
        if not state.is_valid_action(action):
            print(f"{agent_name} made an invalid action!")
            stats["invalid_count"] += 1
            action = random_action
                
        state = state.change_state(action)

    print(f"== {your_agent.__class__.__name__} (1) vs {opponent_agent.__class__.__name__} (2) - First Player: {start_num} ==")

    # To track game result    
    win = 0

    if state.terminal_utility() == 1:
        print("You win!")
        win = 1
    elif state.terminal_utility() == 0:
        print("You lose!")
        win = 0
    else:
        print("Draw")
        win = 0.5

    for agent_name, stats in [("your_agent", your_agent_stats), ("opponent_agent", opponent_agent_stats)]:
        print(f"{agent_name} statistics:")
        print(f"Timeout count: {stats['timeout_count']}")
        print(f"Invalid count: {stats['invalid_count']}")
        
    print(f"Turn count: {turn_count}\n")
    return win

def test_agents(agent, opponent, num_tests):

    wins = 0
    losses = 0
    draws = 0
    
    for i in range(num_tests):
        print(f"Running test {i + 1} (Your Agent as First Player)")
        result = run(agent, opponent, 1) 
        if result == 1:
            wins += 1
        elif result == 0:
            losses += 1
        else:
            draws += 1
            
        print(f"Running test {i + 1} (Your Agent as Second Player)")
        result = run(agent, opponent, 2)  
        if result == 1:
            wins += 1
        elif result == 0:
            losses += 1
        else:
            draws += 1

    # Calculate win rate
    total_games = wins + losses + draws
    win_rate = wins / total_games if total_games > 0 else 0
    loss_rate = losses / total_games if total_games > 0 else 0
    draw_rate = draws / total_games if total_games > 0 else 0

    print("\nOverall Results:")
    print(f"Total games: {total_games}")
    print(f"Wins: {wins} ({win_rate * 100:.2f}%)")
    print(f"Losses: {losses} ({loss_rate * 100:.2f}%)")
    print(f"Draws: {draws} ({draw_rate * 100:.2f}%)")

agent = TenthAgent()  
opponent = NinethAgent()  
test_agents(agent, opponent, 10)
