#!/usr/bin/env python3
import json

# Load one of our debug files to test with real data
with open('debug_llm_response_20250604_215738.json', 'r') as f:
    raw_data = json.load(f)

print("Original reward_functions type:", type(raw_data['reward_functions']))
print("Original reward_functions (first 100 chars):", raw_data['reward_functions'][:100])

# Extract the problematic string
reward_functions_str = raw_data['reward_functions']

print("\n=== Testing Triple Quote Fix ===")

# Method 1: Simple replace triple quotes with single quotes
def fix_method_1(s):
    return s.replace('"""', '"')

# Method 2: More sophisticated - escape the content between triple quotes
import re
def fix_method_2(s):
    def escape_triple_quoted_content(match):
        content = match.group(1)
        # Escape quotes and newlines for JSON
        escaped = content.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')
        return f'"{escaped}"'
    
    # Find """content""" and replace with "escaped_content"
    pattern = r'"""(.*?)"""'
    return re.sub(pattern, escape_triple_quoted_content, s, flags=re.DOTALL)

# Method 3: Even simpler - just remove all triple quotes and their content
def fix_method_3(s):
    pattern = r'""".*?"""'
    return re.sub(pattern, '""', s, flags=re.DOTALL)

# Test each method
for i, fix_func in enumerate([fix_method_1, fix_method_2, fix_method_3], 1):
    print(f"\n--- Method {i} ---")
    try:
        fixed = fix_func(reward_functions_str)
        print(f"Fixed string (first 200 chars): {fixed[:200]}")
        
        # Try to parse it
        parsed = json.loads(fixed)
        print(f"✅ JSON parsing successful! Got {len(parsed)} reward functions")
        
        # Check if reward content is preserved
        if len(parsed) > 0 and 'reward' in parsed[0]:
            reward_length = len(parsed[0]['reward'])
            print(f"First reward function length: {reward_length}")
            if reward_length > 0:
                print(f"First reward preview: {parsed[0]['reward'][:100]}...")
        
    except Exception as e:
        print(f"❌ Method {i} failed: {e}")

print("\n=== Testing on Full Dataset ===")
# Test the best method on the full raw response
best_method = fix_method_2  # Adjust based on results above

try:
    # Copy the original data
    test_data = raw_data.copy()
    
    # Fix the reward_functions
    if isinstance(test_data['reward_functions'], str):
        fixed_str = best_method(test_data['reward_functions'])
        test_data['reward_functions'] = json.loads(fixed_str)
        
        print(f"✅ Full test successful!")
        print(f"Reward functions count: {len(test_data['reward_functions'])}")
        for i, rf in enumerate(test_data['reward_functions']):
            print(f"  {i+1}. {rf['name']} ({rf['type']}) - {len(rf['reward'])} chars")
            
except Exception as e:
    print(f"❌ Full test failed: {e}")
