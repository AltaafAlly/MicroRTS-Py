#!/usr/bin/env python3
"""
Simple Cluster Test for MicroRTS Tournament
===========================================

A minimal test to verify that the basic MicroRTS setup works on the cluster.
This runs a single match to test the environment before running the full tournament.
"""

import os
import sys
import json
from pathlib import Path

def test_basic_imports():
    """Test that we can import the required modules"""
    print("Testing basic imports...")
    try:
        import gym_microrts
        print("✅ gym_microrts imported successfully")
        
        from gym_microrts.envs.vec_env import MicroRTSBotVecEnv
        print("✅ MicroRTSBotVecEnv imported successfully")
        
        from gym_microrts import microrts_ai
        print("✅ microrts_ai imported successfully")
        
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_jvm_startup():
    """Test that the JVM starts correctly"""
    print("\nTesting JVM startup...")
    try:
        from gym_microrts.envs.vec_env import MicroRTSBotVecEnv
        
        # Try to create a simple environment
        env = MicroRTSBotVecEnv(
            ai1s=[microrts_ai.workerRushAI],
            ai2s=[microrts_ai.lightRushAI],
            max_steps=100,  # Very short test
            map_paths="maps/8x8/basesWorkers8x8A.xml"
        )
        print("✅ JVM started and environment created successfully")
        
        # Test a simple reset
        obs = env.reset()
        print("✅ Environment reset successful")
        
        # Test a simple step
        actions = [0, 0]  # Simple actions
        obs, reward, done, info = env.step(actions)
        print("✅ Environment step successful")
        
        env.close()
        print("✅ Environment closed successfully")
        
        return True
    except Exception as e:
        print(f"❌ JVM/Environment test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_utt_loading():
    """Test that custom UTTs can be loaded"""
    print("\nTesting UTT loading...")
    try:
        from gym_microrts.envs.vec_env import MicroRTSBotVecEnv
        
        # Test with custom UTT
        env = MicroRTSBotVecEnv(
            ai1s=[microrts_ai.workerRushAI],
            ai2s=[microrts_ai.lightRushAI],
            max_steps=50,
            map_paths="maps/8x8/basesWorkers8x8A.xml",
            utt_json_p0="utts/CustomDemoUTT.json",
            utt_json_p1="utts/CustomDemoUTT.json"
        )
        print("✅ Custom UTT loaded successfully")
        
        obs = env.reset()
        print("✅ Environment with custom UTT reset successful")
        
        env.close()
        print("✅ Custom UTT test completed")
        
        return True
    except Exception as e:
        print(f"❌ UTT loading test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_single_match():
    """Run a single match to test the full pipeline"""
    print("\nRunning single match test...")
    try:
        from gym_microrts.envs.vec_env import MicroRTSBotVecEnv
        from gym_microrts import microrts_ai
        
        # Create environment
        env = MicroRTSBotVecEnv(
            ai1s=[microrts_ai.workerRushAI],
            ai2s=[microrts_ai.lightRushAI],
            max_steps=200,  # Short match
            map_paths="maps/8x8/basesWorkers8x8A.xml"
        )
        
        # Run a short match
        obs = env.reset()
        done = False
        steps = 0
        
        while not done and steps < 200:
            actions = [0, 0]  # Simple actions
            obs, reward, done, info = env.step(actions)
            steps += 1
            
            if steps % 50 == 0:
                print(f"  Step {steps}: reward={reward}, done={done}")
        
        print(f"✅ Single match completed in {steps} steps")
        print(f"   Final reward: {reward}")
        print(f"   Match result: {'Done' if done else 'Max steps reached'}")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"❌ Single match test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("MICRORTS CLUSTER TEST")
    print("=" * 60)
    print(f"Python version: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    print(f"Java version:")
    os.system("java -version 2>&1 | head -1")
    print()
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("JVM Startup", test_jvm_startup),
        ("UTT Loading", test_utt_loading),
        ("Single Match", run_single_match),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:20} {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 ALL TESTS PASSED! The cluster setup is working.")
        print("You can now run the full tournament.")
    else:
        print("⚠️  SOME TESTS FAILED. Check the errors above.")
        print("Fix the issues before running the full tournament.")
    print("=" * 60)
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
