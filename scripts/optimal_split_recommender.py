#!/usr/bin/env python3
"""
Optimal Split Option Recommender
================================

This script helps users choose the optimal dataset split option
based on their specific use case and requirements.

Author: ASV Dataset Preparation System
"""

import sys
from pathlib import Path
from typing import Dict, List, Any

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))


class OptimalSplitRecommender:
    """Recommender system for optimal split options"""
    
    def __init__(self):
        self.recommendations = {
            'A': {
                'name': 'Original Splits',
                'best_for': [
                    'Reproducing published results',
                    'Comparing with existing research',
                    'Maintaining dataset integrity',
                    'Quick testing and validation'
                ],
                'pros': [
                    'Preserves original dataset structure',
                    'Ensures reproducibility',
                    'Fast processing',
                    'Standard benchmark setup'
                ],
                'cons': [
                    'Unbalanced user distribution',
                    'Limited cross-user validation',
                    'Some users only in test set'
                ],
                'use_cases': [
                    'academic_research',
                    'benchmarking',
                    'quick_testing',
                    'reproducibility'
                ]
            },
            'B': {
                'name': 'Combined Redistribution',
                'best_for': [
                    'Robust model training',
                    'Cross-user generalization',
                    'Balanced evaluation',
                    'Production deployment'
                ],
                'pros': [
                    'All users in all splits',
                    'Balanced distribution',
                    'Better generalization',
                    'Comprehensive evaluation'
                ],
                'cons': [
                    'Longer processing time',
                    'May not match published results',
                    'More complex validation'
                ],
                'use_cases': [
                    'production_training',
                    'robust_evaluation',
                    'cross_validation',
                    'generalization_testing'
                ]
            },
            'C': {
                'name': 'Train+Dev vs Eval',
                'best_for': [
                    'Maximum training data',
                    'Simple train/test setup',
                    'Resource-constrained scenarios',
                    'Initial model development'
                ],
                'pros': [
                    'Maximum training data',
                    'Simple setup',
                    'Fast validation',
                    'Good for initial experiments'
                ],
                'cons': [
                    'No separate validation set',
                    'Risk of overfitting',
                    'Limited evaluation strategy'
                ],
                'use_cases': [
                    'initial_development',
                    'data_maximization',
                    'simple_evaluation',
                    'proof_of_concept'
                ]
            }
        }
    
    def display_interactive_questionnaire(self):
        """Interactive questionnaire to recommend optimal split"""
        print("🎯 OPTIMAL SPLIT OPTION RECOMMENDER")
        print("="*50)
        print("Answer a few questions to get personalized recommendations!")
        print()
        
        # Collect user preferences
        answers = {}
        
        # Question 1: Primary use case
        print("1️⃣ What is your primary use case?")
        print("   A) Academic research / benchmarking")
        print("   B) Production model development")
        print("   C) Initial experimentation / proof of concept")
        print("   D) Comparing with existing research")
        
        while True:
            choice = input("\nEnter A, B, C, or D: ").strip().upper()
            if choice in ['A', 'B', 'C', 'D']:
                answers['use_case'] = choice
                break
            print("Please enter A, B, C, or D")
        
        # Question 2: Training data priority
        print("\n2️⃣ How important is maximizing training data?")
        print("   A) Very important - need as much data as possible")
        print("   B) Balanced approach - quality over quantity")
        print("   C) Not critical - standard splits are fine")
        
        while True:
            choice = input("\nEnter A, B, or C: ").strip().upper()
            if choice in ['A', 'B', 'C']:
                answers['data_priority'] = choice
                break
            print("Please enter A, B, or C")
        
        # Question 3: Evaluation strategy
        print("\n3️⃣ What evaluation strategy do you prefer?")
        print("   A) Simple train/test evaluation")
        print("   B) Comprehensive train/dev/test evaluation")
        print("   C) Cross-user generalization testing")
        
        while True:
            choice = input("\nEnter A, B, or C: ").strip().upper()
            if choice in ['A', 'B', 'C']:
                answers['evaluation'] = choice
                break
            print("Please enter A, B, or C")
        
        # Question 4: Reproducibility importance
        print("\n4️⃣ How important is reproducing published results?")
        print("   A) Very important - must match exactly")
        print("   B) Somewhat important - similar results acceptable")
        print("   C) Not important - new approach is fine")
        
        while True:
            choice = input("\nEnter A, B, or C: ").strip().upper()
            if choice in ['A', 'B', 'C']:
                answers['reproducibility'] = choice
                break
            print("Please enter A, B, or C")
        
        return self.generate_recommendation(answers)
    
    def generate_recommendation(self, answers: Dict[str, str]) -> str:
        """Generate recommendation based on user answers"""
        scores = {'A': 0, 'B': 0, 'C': 0}
        
        # Score based on use case
        if answers['use_case'] == 'A':  # Academic research
            scores['A'] += 3
            scores['B'] += 1
        elif answers['use_case'] == 'B':  # Production
            scores['B'] += 3
            scores['C'] += 1
        elif answers['use_case'] == 'C':  # Initial experimentation
            scores['C'] += 3
            scores['A'] += 1
        elif answers['use_case'] == 'D':  # Comparing with research
            scores['A'] += 3
        
        # Score based on data priority
        if answers['data_priority'] == 'A':  # Maximize data
            scores['C'] += 2
            scores['B'] += 1
        elif answers['data_priority'] == 'B':  # Balanced
            scores['B'] += 2
            scores['A'] += 1
        elif answers['data_priority'] == 'C':  # Standard
            scores['A'] += 2
        
        # Score based on evaluation strategy
        if answers['evaluation'] == 'A':  # Simple
            scores['C'] += 2
            scores['A'] += 1
        elif answers['evaluation'] == 'B':  # Comprehensive
            scores['A'] += 2
            scores['B'] += 1
        elif answers['evaluation'] == 'C':  # Cross-user
            scores['B'] += 3
        
        # Score based on reproducibility
        if answers['reproducibility'] == 'A':  # Very important
            scores['A'] += 3
        elif answers['reproducibility'] == 'B':  # Somewhat important
            scores['A'] += 1
            scores['B'] += 1
        elif answers['reproducibility'] == 'C':  # Not important
            scores['B'] += 1
            scores['C'] += 1
        
        # Find best option
        best_option = max(scores, key=scores.get)
        return best_option
    
    def display_recommendation(self, recommended_option: str, answers: Dict[str, str]):
        """Display the recommendation with detailed explanation"""
        option_info = self.recommendations[recommended_option]
        
        print("\n" + "="*60)
        print("🎯 RECOMMENDATION RESULTS")
        print("="*60)
        
        print(f"\n🏆 RECOMMENDED OPTION: {recommended_option}")
        print(f"📝 {option_info['name']}")
        print()
        
        print("✅ WHY THIS OPTION:")
        for reason in option_info['best_for']:
            print(f"   • {reason}")
        print()
        
        print("👍 PROS:")
        for pro in option_info['pros']:
            print(f"   • {pro}")
        print()
        
        print("👎 CONS:")
        for con in option_info['cons']:
            print(f"   • {con}")
        print()
        
        # Show alternatives
        print("🔄 ALTERNATIVE OPTIONS:")
        for opt in ['A', 'B', 'C']:
            if opt != recommended_option:
                alt_info = self.recommendations[opt]
                print(f"   Option {opt}: {alt_info['name']}")
                print(f"     Best for: {', '.join(alt_info['best_for'][:2])}")
        
        print("\n" + "="*60)
        print("💡 NEXT STEPS:")
        print(f"   Run: python asv_cli.py convert --split-option {recommended_option}")
        print("   Or: python asv_cli.py convert (for interactive mode)")
        print("="*60)
    
    def display_all_options_comparison(self):
        """Display comparison of all options"""
        print("\n📊 COMPLETE OPTIONS COMPARISON")
        print("="*70)
        
        for option, info in self.recommendations.items():
            print(f"\n📋 OPTION {option}: {info['name']}")
            print("─" * 50)
            print("🎯 Best for:")
            for item in info['best_for']:
                print(f"   • {item}")
            print("👍 Pros:")
            for item in info['pros']:
                print(f"   • {item}")
            print("👎 Cons:")
            for item in info['cons']:
                print(f"   • {item}")
    
    def run_recommender(self):
        """Run the complete recommendation system"""
        print("Welcome to the ASV Dataset Split Option Recommender!")
        print()
        
        while True:
            print("Choose an option:")
            print("1) Get personalized recommendation (questionnaire)")
            print("2) Compare all options")
            print("3) Exit")
            
            choice = input("\nEnter 1, 2, or 3: ").strip()
            
            if choice == '1':
                recommended_option = self.display_interactive_questionnaire()
                answers = {}  # Would need to store answers from questionnaire
                self.display_recommendation(recommended_option, answers)
                break
            elif choice == '2':
                self.display_all_options_comparison()
                break
            elif choice == '3':
                print("Goodbye!")
                break
            else:
                print("Please enter 1, 2, or 3")


def main():
    """Main execution function"""
    recommender = OptimalSplitRecommender()
    recommender.run_recommender()
    return 0


if __name__ == "__main__":
    sys.exit(main()) 