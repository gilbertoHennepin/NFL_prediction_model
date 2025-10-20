# NFL Game Prediction using nfl_data_py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot  
import warnings
import os
warnings.filterwarnings('ignore')

# Core ML libraries
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedStratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.calibration import CalibratedClassifierCV as CCV
from sklearn.pipeline import Pipeline

# Advanced ML libraries
try: 
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("XGBoost not available. Install with: pip install xgboost")

# NFL data library
try:
    import nfl_data_py as nfl
    HAS_NFL_DATA = True
except ImportError:
    HAS_NFL_DATA = False
    print("nfl_data_py not available. Install with: pip install nfl_data_py")

class NFLGamePredictor:
    def __init__(self):
        self.models = {}
        self.best_features = []
        self.scaler = StandardScaler()
        self.final_model = None
        
    def collect_data(self, start_year=2010, end_year=2024, save_csv=True, data_folder='nfl_data'):
        """Collect comprehensive NFL data using nfl_data_py and save as CSV files"""
        
        if not HAS_NFL_DATA:
            raise ImportError("nfl_data_py is required. Install with: pip install nfl_data_py")
        
        print(f"Collecting NFL data from {start_year} to {end_year}...")
        
        # Create data folder if it doesn't exist
        if save_csv and not os.path.exists(data_folder):
            os.makedirs(data_folder)
            print(f"Created data folder: {data_folder}")
        
        years = list(range(start_year, end_year + 1))
        
        # Get play-by-play data for game-level statistics
        print("Downloading play-by-play data...")
        try:
            pbp_data = nfl.import_pbp_data(years)
            print(f"✓ Play-by-play data downloaded successfully")
        except Exception as e:
            print(f"Error downloading play-by-play data: {e}")
            # Try with reduced year range
            years = list(range(start_year, 2024 + 1))
            pbp_data = nfl.import_pbp_data(years)
            print(f"✓ Play-by-play data downloaded (adjusted to {start_year}-2024)")
        
        # Get team information
        print("Downloading team information...")
        teams = nfl.import_team_desc()
        team_dict = teams.set_index('team_abbr')['team_name'].to_dict()
        
        # Get weekly data for season-long team statistics
        print("Downloading weekly team data...")
        try:
            weekly_data = nfl.import_weekly_data(years)
            print(f"✓ Weekly data downloaded successfully")
        except Exception as e:
            print(f"Error downloading weekly data: {e}")
            print("Trying with 2015-2024 data only...")
            # Fall back to confirmed available years
            years_safe = list(range(2015, 2024 + 1))
            weekly_data = nfl.import_weekly_data(years_safe)
            print(f"✓ Weekly data downloaded (2015-2024)")
        
        # Get schedule data
        print("Downloading schedule data...")
        try:
            schedule_data = nfl.import_schedules(years)
            print(f"✓ Schedule data downloaded successfully")
        except Exception as e:
            print(f"Error downloading schedule data: {e}")
            print("Trying with 2015-2024 data only...")
            # Fall back to confirmed available years
            years_safe = list(range(2015, 2024 + 1))
            schedule_data = nfl.import_schedules(years_safe)
            print(f"✓ Schedule data downloaded (2015-2024)")
        
        # Try to get 2025 Week 1 data separately if available
        print("Attempting to download 2025 Week 1 data...")
        try:
            pbp_2025 = nfl.import_pbp_data([2025])
            if not pbp_2025.empty:
                pbp_data = pd.concat([pbp_data, pbp_2025], ignore_index=True)
                print("✓ 2025 play-by-play data added")
            
            weekly_2025 = nfl.import_weekly_data([2025])
            if not weekly_2025.empty:
                weekly_data = pd.concat([weekly_data, weekly_2025], ignore_index=True)
                print("✓ 2025 weekly data added")
                
            schedule_2025 = nfl.import_schedules([2025])
            if not schedule_2025.empty:
                schedule_data = pd.concat([schedule_data, schedule_2025], ignore_index=True)
                print("✓ 2025 schedule data added")
        except Exception as e:
            print(f"2025 data not available yet: {e}")
            print("Proceeding with 2015-2024 data only")
        
        print(f"Collected {len(pbp_data)} play-by-play records")
        print(f"Collected {len(weekly_data)} weekly team records") 
        print(f"Collected {len(schedule_data)} scheduled games")
        
        # Save data as CSV files if requested
        if save_csv:
            print(f"\nSaving data to CSV files in '{data_folder}' folder...")
            
            # Save play-by-play data (this will be large)
            pbp_file = os.path.join(data_folder, f'pbp_data_{start_year}_{end_year}.csv')
            pbp_data.to_csv(pbp_file, index=False)
            print(f"✓ Saved play-by-play data: {pbp_file} ({len(pbp_data):,} rows)")
            
            # Save weekly data
            weekly_file = os.path.join(data_folder, f'weekly_data_{start_year}_{end_year}.csv')
            weekly_data.to_csv(weekly_file, index=False)
            print(f"✓ Saved weekly data: {weekly_file} ({len(weekly_data):,} rows)")
            
            # Save schedule data
            schedule_file = os.path.join(data_folder, f'schedule_data_{start_year}_{end_year}.csv')
            schedule_data.to_csv(schedule_file, index=False)
            print(f"✓ Saved schedule data: {schedule_file} ({len(schedule_data):,} rows)")
            
            # Save team information
            teams_file = os.path.join(data_folder, 'team_info.csv')
            teams.to_csv(teams_file, index=False)
            print(f"✓ Saved team info: {teams_file} ({len(teams):,} rows)")
            
            # Save a sample of each dataset for quick inspection
            print(f"\nSaving sample data for quick inspection...")
            
            # Sample play-by-play (first 1000 rows)
            pbp_sample_file = os.path.join(data_folder, 'pbp_sample.csv')
            pbp_data.head(1000).to_csv(pbp_sample_file, index=False)
            print(f"✓ Saved PBP sample: {pbp_sample_file} (1,000 rows)")
            
            # Sample weekly data (recent season)
            weekly_sample = weekly_data[weekly_data['season'] >= end_year - 1]
            weekly_sample_file = os.path.join(data_folder, 'weekly_sample.csv')
            weekly_sample.to_csv(weekly_sample_file, index=False)
            print(f"✓ Saved weekly sample: {weekly_sample_file} ({len(weekly_sample):,} rows from {end_year-1}-{end_year})")
            
            # Show column information
            print(f"\nDATA STRUCTURE OVERVIEW:")
            print("="*50)
            print(f"Play-by-Play Columns ({len(pbp_data.columns)}): {list(pbp_data.columns[:10])}...")
            print(f"Weekly Data Columns ({len(weekly_data.columns)}): {list(weekly_data.columns)}")
            print(f"Schedule Columns ({len(schedule_data.columns)}): {list(schedule_data.columns)}")
            print(f"Team Info Columns ({len(teams.columns)}): {list(teams.columns)}")
            
            print(f"\nDATA SAVED SUCCESSFULLY!")
            print(f"Check the '{data_folder}' folder to examine the downloaded data.")
        
        return pbp_data, weekly_data, schedule_data, team_dict
    
    def _calculate_injury_percentage(self, team_data):
        """
        Calculate estimated injury percentage based on available performance metrics
        
        This method estimates team injury impact by analyzing performance consistency
        and key player availability indicators in the data.
        """
        
        # If we have specific injury data columns, use them
        if 'injuries' in team_data.columns:
            return team_data['injuries'].mean()
        
        # Estimate injury impact based on performance variance and available metrics
        # Higher variance in key stats might indicate injury-related inconsistency
        
        injury_indicators = []
        
        # 1. Passing performance consistency (QB health indicator)
        if 'passing_yards' in team_data.columns and len(team_data) > 1:
            passing_std = team_data['passing_yards'].std()
            passing_mean = team_data['passing_yards'].mean()
            if passing_mean > 0:
                passing_variance = (passing_std / passing_mean) * 100
                injury_indicators.append(min(passing_variance, 30))  # Cap at 30%
        
        # 2. Rushing performance consistency (RB/OL health indicator)
        if 'rushing_yards' in team_data.columns and len(team_data) > 1:
            rushing_std = team_data['rushing_yards'].std()
            rushing_mean = team_data['rushing_yards'].mean()
            if rushing_mean > 0:
                rushing_variance = (rushing_std / rushing_mean) * 100
                injury_indicators.append(min(rushing_variance, 25))  # Cap at 25%
        
        # 3. Completion percentage consistency (QB/WR health indicator)
        if 'completions' in team_data.columns and 'passing_attempts' in team_data.columns:
            comp_pct = team_data['completions'] / (team_data['passing_attempts'] + 0.1)  # Avoid division by zero
            if len(comp_pct) > 1:
                comp_std = comp_pct.std()
                comp_variance = comp_std * 100
                injury_indicators.append(min(comp_variance, 20))  # Cap at 20%
        
        # 4. Fantasy points consistency (overall team health)
        if 'fantasy_points' in team_data.columns and len(team_data) > 1:
            fp_std = team_data['fantasy_points'].std()
            fp_mean = team_data['fantasy_points'].mean()
            if fp_mean > 0:
                fp_variance = (fp_std / fp_mean) * 100
                injury_indicators.append(min(fp_variance, 35))  # Cap at 35%
        
        # Calculate weighted average injury percentage
        if injury_indicators:
            # Weight more recent games higher (if we have game order info)
            weights = [1.0] * len(injury_indicators)  
            weighted_avg = sum(i * w for i, w in zip(injury_indicators, weights)) / sum(weights)
            
            # Normalize to 0-100% range and apply league baseline
            # NFL teams typically have 10-25% of roster dealing with some injury
            baseline_injury_rate = 15.0  # League average baseline
            estimated_injury_pct = min(max(weighted_avg, 5.0), 40.0)  # 5-40% range
            
            # Blend with baseline for more realistic estimates
            final_injury_pct = (estimated_injury_pct * 0.7) + (baseline_injury_rate * 0.3)
            
            return final_injury_pct
        
        # Default injury rate if no data available
        return 15.0  # NFL league average
    
    def create_team_features(self, weekly_data, season, week):
        """Create team-level features for a specific season/week"""
        
        # Filter data up to the current week
        season_data = weekly_data[
            (weekly_data['season'] == season) & 
            (weekly_data['week'] < week)
        ]
        
        if season_data.empty:
            return {}
        
        # Calculate season averages for each team
        team_features = {}
        
        for team in season_data['recent_team'].unique():
            team_data = season_data[season_data['recent_team'] == team]
            
            if len(team_data) == 0:
                continue
                
            # Offensive features
            features = {
                'passing_yards_pg': team_data['passing_yards'].mean(),
                'rushing_yards_pg': team_data['rushing_yards'].mean(), 
                'total_yards_pg': team_data['passing_yards'].mean() + team_data['rushing_yards'].mean(),
                'points_pg': team_data['fantasy_points'].mean() if 'fantasy_points' in team_data.columns else 0,
                'completions_pg': team_data['completions'].mean(),
                'passing_tds_pg': team_data['passing_tds'].mean(),
                'interceptions_thrown_pg': team_data['interceptions'].mean(),
                'rushing_tds_pg': team_data['rushing_tds'].mean(),
                'fumbles_lost_pg': team_data['fumbles_lost'].mean() if 'fumbles_lost' in team_data.columns else 0,
                
                # Defensive features (opponent stats)
                'opp_passing_yards_pg': 0,  # Will be calculated separately
                'opp_rushing_yards_pg': 0,
                'opp_points_pg': 0,
                
                # Team health and availability metrics
                'injury_percentage': self._calculate_injury_percentage(team_data),
                
                # Advanced metrics
                'turnover_ratio': (team_data['interceptions'].mean() - 
                                 team_data['interceptions'].mean()),  # Simplified
                'games_played': len(team_data)
            }
            
            team_features[team] = features
        
        return team_features
    
    def create_game_features(self, home_team, away_team, team_features, 
                           season, week, is_playoff=False, is_neutral=False):
        """Create features for a specific matchup"""
        
        if home_team not in team_features or away_team not in team_features:
            return None
        
        home_stats = team_features[home_team]
        away_stats = team_features[away_team]
        
        # Create matchup features
        features = {
            # Home team offensive stats
            'home_passing_ypg': home_stats['passing_yards_pg'],
            'home_rushing_ypg': home_stats['rushing_yards_pg'],
            'home_total_ypg': home_stats['total_yards_pg'],
            'home_points_pg': home_stats['points_pg'],
            'home_passing_tds_pg': home_stats['passing_tds_pg'],
            'home_turnovers_pg': home_stats.get('fumbles_lost_pg', 0) + home_stats['interceptions_thrown_pg'],
            'home_injury_pct': home_stats['injury_percentage'],
            
            # Away team offensive stats  
            'away_passing_ypg': away_stats['passing_yards_pg'],
            'away_rushing_ypg': away_stats['rushing_yards_pg'],
            'away_total_ypg': away_stats['total_yards_pg'],
            'away_points_pg': away_stats['points_pg'],
            'away_passing_tds_pg': away_stats['passing_tds_pg'],
            'away_turnovers_pg': away_stats.get('fumbles_lost_pg', 0) + away_stats['interceptions_thrown_pg'],
            'away_injury_pct': away_stats['injury_percentage'],
            
            # Matchup advantages
            'passing_advantage': home_stats['passing_yards_pg'] - away_stats['passing_yards_pg'],
            'rushing_advantage': home_stats['rushing_yards_pg'] - away_stats['rushing_yards_pg'],
            'scoring_advantage': home_stats['points_pg'] - away_stats['points_pg'],
            'turnover_advantage': away_stats.get('fumbles_lost_pg', 0) + away_stats['interceptions_thrown_pg'] - 
                                (home_stats.get('fumbles_lost_pg', 0) + home_stats['interceptions_thrown_pg']),
            'injury_advantage': away_stats['injury_percentage'] - home_stats['injury_percentage'],  # Lower injury % is better
            
            # Game context
            'home_field_advantage': 0 if is_neutral else 2.5,
            'is_playoff': 1 if is_playoff else 0,
            'is_neutral': 1 if is_neutral else 0,
            'week': week,
            'season': season,
        }
        
        return features
    
    def build_dataset(self, pbp_data, weekly_data, schedule_data, save_csv=True, data_folder='nfl_data'):
        """Build complete dataset from NFL data and optionally save as CSV"""
        
        print("Building dataset from collected data...")
        
        game_records = []
        
        # Process each scheduled game
        for _, game in schedule_data.iterrows():
            season = game['season']
            week = game['week']
            home_team = game['home_team']
            away_team = game['away_team']
            
            # Skip if missing essential data
            if pd.isna(home_team) or pd.isna(away_team):
                continue
            
            # Get team features up to this point in season
            team_features = self.create_team_features(weekly_data, season, week)
            
            if not team_features:
                continue
            
            # Create game features
            game_features = self.create_game_features(
                home_team, away_team, team_features, 
                season, week,
                is_playoff=game.get('game_type', '') == 'REG',
                is_neutral=False  # Simplified for now
            )
            
            if game_features is None:
                continue
            
            # Determine result (home team win = 1, loss = 0)
            home_score = game.get('home_score', 0)
            away_score = game.get('away_score', 0)
            
            # Skip games without scores (future games)
            if pd.isna(home_score) or pd.isna(away_score):
                continue
            
            game_features['home_win'] = 1 if home_score > away_score else 0
            game_features['home_score'] = home_score
            game_features['away_score'] = away_score
            game_features['game_id'] = f"{season}_{week}_{home_team}_{away_team}"
            
            game_records.append(game_features)
        
        df = pd.DataFrame(game_records)
        print(f"Created dataset with {len(df)} games")
        
        # Save the processed dataset
        if save_csv and not df.empty:
            if not os.path.exists(data_folder):
                os.makedirs(data_folder)
            
            processed_file = os.path.join(data_folder, 'processed_game_features.csv')
            df.to_csv(processed_file, index=False)
            print(f"✓ Saved processed game features: {processed_file} ({len(df):,} rows)")
            
            # Show feature information
            print(f"\nPROCESSED FEATURES ({len(df.columns)} columns):")
            print("="*50)
            for col in df.columns:
                print(f"  - {col}")
        
        return df
    
    def select_features(self, df, n_features=13):
        """Select best features using RFE"""
        
        # Prepare data
        feature_cols = [col for col in df.columns 
                       if col not in ['home_win', 'home_score', 'away_score', 'game_id']]
        
        X = df[feature_cols]
        y = df['home_win']
        
        # Remove any columns with all NaN or constant values
        X = X.loc[:, X.var() > 0]
        X = X.fillna(X.mean())
        
        print(f"Starting feature selection with {len(X.columns)} features...")
        
        # Use RFE with different numbers of features
        models = {}
        results = []
        
        for i in range(2, min(n_features + 1, len(X.columns) + 1)):
            rfe = RFE(estimator=LDA(), n_features_to_select=i)
            model = DecisionTreeClassifier(random_state=42)
            pipeline = Pipeline(steps=[('s', rfe), ('m', model)])
            
            cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)
            scores = cross_val_score(pipeline, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
            
            results.append(scores)
            models[str(i)] = pipeline
            
            print(f'{i} features: {scores.mean():.3f} (+/- {scores.std():.3f})')
        
        # Find best number of features
        best_idx = np.argmax([np.mean(result) for result in results])
        best_n_features = best_idx + 2
        
        print(f"Best number of features: {best_n_features}")
        
        # Get the best feature set
        rfe = RFE(estimator=LDA(), n_features_to_select=best_n_features)
        rfe.fit(X, y)
        
        selected_features = X.columns[rfe.support_].tolist()
        self.best_features = selected_features
        
        print("Selected features:")
        for feature in selected_features:
            print(f"  - {feature}")
        
        return selected_features
    
    def train_models(self, df):
        """Train and compare multiple models"""
        
        if not self.best_features:
            self.select_features(df)
        
        # Prepare data
        X = df[self.best_features].fillna(df[self.best_features].mean())
        y = df['home_win']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Define models to test
        models_to_test = {
            'Logistic Regression': LogisticRegression(random_state=42),
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'Random Forest': RandomForestClassifier(random_state=42)
        }
        
        if HAS_XGB:
            models_to_test['XGBoost'] = xgb.XGBClassifier(random_state=42, verbosity=0)
        
        # Train and evaluate models
        model_results = {}
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)
        
        print("\nTraining and evaluating models...")
        
        for name, model in models_to_test.items():
            # Use scaled data for logistic regression, raw for tree-based
            X_use = X_train_scaled if 'Logistic' in name else X_train
            scores = cross_val_score(model, X_use, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
            
            model_results[name] = {
                'mean_score': scores.mean(),
                'std_score': scores.std(),
                'scores': scores
            }
            
            print(f"{name}: {scores.mean():.3f} (+/- {scores.std():.3f})")
        
        self.models = models_to_test
        return model_results
    
    def tune_best_model(self, df, model_name='Random Forest'):
        """Tune hyperparameters for the best performing model"""
        
        X = df[self.best_features].fillna(df[self.best_features].mean())
        y = df['home_win']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        if model_name == 'Random Forest':
            param_grid = {
                'n_estimators': [200, 300, 400, 500],
                'max_depth': [8, 9, 10, 11, 12],
                'min_samples_leaf': [2, 5],
                'criterion': ['gini', 'entropy']
            }
            model = RandomForestClassifier(random_state=42)
        
        elif model_name == 'Logistic Regression':
            param_grid = {
                'C': [0.01, 0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear']
            }
            model = LogisticRegression(random_state=42)
            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)
        
        elif model_name == 'XGBoost' and HAS_XGB:
            param_grid = {
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
            'n_estimators': [100, 200, 300],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0],
            'reg_alpha': [0, 1],
            'reg_lambda': [1, 5]
            }
            model = xgb.XGBClassifier(random_state=42, verbosity=0)
        
        else:
            print(f"Tuning not implemented for {model_name}")
            return None
        
        print(f"\nTuning {model_name}...")
        
        # Grid search
        grid_search = GridSearchCV(
            model, param_grid, 
            cv=5, scoring='accuracy', 
            n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.3f}")
        
        # Test on holdout set
        test_score = grid_search.score(X_test, y_test)
        print(f"Test set accuracy: {test_score:.3f}")
        
        return grid_search.best_estimator_
    
    def create_ensemble_model(self, df):
        """Create ensemble model combining multiple algorithms"""
        
        X = df[self.best_features].fillna(df[self.best_features].mean())
        y = df['home_win']
        
        # Individual models
        rf = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)
        lr = LogisticRegression(C=1, random_state=42)
        
        estimators = [('rf', rf), ('lr', lr)]
        
        if HAS_XGB:
            xgb_model = xgb.XGBClassifier(max_depth=5, learning_rate=0.1, n_estimators=200, 
                                        random_state=42, verbosity=0)
            estimators.append(('xgb', xgb_model))
        
        # Voting classifier
        voting_clf = VotingClassifier(estimators=estimators, voting='soft')
        
        # Calibrated classifier for better probability estimates
        self.final_model = CCV(voting_clf, method='isotonic', cv=3)
        
        print("Training ensemble model...")
        self.final_model.fit(X, y)
        
        return self.final_model
    
    def predict_games(self, games_df, confidence_threshold=0.6):
        """Make predictions on new games"""
        
        if self.final_model is None:
            raise ValueError("Model not trained yet. Call create_ensemble_model() first.")
        
        X = games_df[self.best_features].fillna(games_df[self.best_features].mean())
        
        # Get probability predictions
        probabilities = self.final_model.predict_proba(X)[:, 1]
        predictions = self.final_model.predict(X)
        
        # Create results dataframe
        results = games_df.copy()
        results['home_win_prob'] = probabilities
        results['predicted_home_win'] = predictions
        
        # High-confidence bets
        results['high_confidence_bet'] = (
            (probabilities >= confidence_threshold) | 
            (probabilities <= (1 - confidence_threshold))
        )
        
        return results

# Initialize predictor
predictor = NFLGamePredictor()

# Check if nfl_data_py is available
if not HAS_NFL_DATA:
    print("Please install nfl_data_py to use this predictor:")
    print("pip install nfl_data_py")
    exit()

try:
    # Collect data (this may take a few minutes)
    print("This may take a few minutes to download NFL data...")
    pbp_data, weekly_data, schedule_data, team_dict = predictor.collect_data(2015, 2025, save_csv=True)
    
    # Build dataset
    df = predictor.build_dataset(pbp_data, weekly_data, schedule_data, save_csv=True)
    
    if df.empty:
        print("No data collected. Check your internet connection and try again.")
        exit()
    
    # Feature selection
    predictor.select_features(df, n_features=13)
    
    # Train models
    model_results = predictor.train_models(df)
    
    # Create ensemble
    final_model = predictor.create_ensemble_model(df)
    
    # Example prediction on test set
    train_data = df[df['season'] < 2023]
    test_data = df[df['season'] == 2023]
    
    if not test_data.empty:
        predictor.final_model.fit(
            train_data[predictor.best_features].fillna(train_data[predictor.best_features].mean()), 
            train_data['home_win']
        )
        
        predictions = predictor.predict_games(test_data)
        
        # Calculate accuracy
        accuracy = (predictions['predicted_home_win'] == predictions['home_win']).mean()
        print(f"\n2023 season prediction accuracy: {accuracy:.3f}")
        
        # High confidence bets
        high_conf = predictions[predictions['high_confidence_bet']]
        if not high_conf.empty:
            conf_accuracy = (high_conf['predicted_home_win'] == high_conf['home_win']).mean()
            print(f"High confidence bet accuracy: {conf_accuracy:.3f} ({len(high_conf)} games)")
    
    print("\nPredictor trained successfully!")
    print("You can now use predictor.predict_games() on new data.")
    
except Exception as e:
    print(f"Error: {e}")
    print("Make sure you have a stable internet connection for data download.")
