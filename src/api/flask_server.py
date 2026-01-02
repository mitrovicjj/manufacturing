from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import os
import json
import pandas as pd

# Error handling za import
try:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
    from src.sim_integration.tecnomatix_bridge import TecnomatixBridge
    print("✅ TecnomatixBridge imported successfully")
except Exception as e:
    print(f"Error importing TecnomatixBridge: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

app = Flask(__name__)
CORS(app)

# Initialize bridge
try:
    print("🔄 Initializing TecnomatixBridge...")
    bridge = TecnomatixBridge()
    print("✅ Bridge initialized successfully")
except Exception as e:
    print(f"❌ Error initializing bridge: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "ok", 
        "service": "Manufacturing ML API",
        "model_loaded": True
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Single cycle prediction."""
    try:
        data = request.get_json()
        result_json = bridge.predict_json(json.dumps(data))
        return jsonify(json.loads(result_json))
    except Exception as e:
        return jsonify({"error": str(e), "success": False}), 400

@app.route('/predict_with_history', methods=['POST'])
def predict_with_history():
    """Prediction with historical context (RECOMMENDED)."""
    try:
        data = request.get_json()
        current = data.get('current')
        history = data.get('history', [])
        
        if not current:
            return jsonify({"error": "Missing 'current' cycle data"}), 400
        
        from src.ml.predict_production import ProductionPredictor
        predictor = ProductionPredictor()
        
        result = predictor.predict_with_history(current, history)
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": str(e), "success": False}), 400

@app.route('/should_maintain', methods=['POST'])
def should_maintain():
    """Simple boolean endpoint for Tecnomatix decision logic."""
    try:
        data = request.get_json()
        trigger = bridge.should_trigger_maintenance(json.dumps(data))
        return jsonify({
            "should_trigger_maintenance": trigger,
            "timestamp": pd.Timestamp.now().isoformat()
        })
    except Exception as e:
        return jsonify({"error": str(e), "success": False}), 400

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """Batch prediction endpoint."""
    try:
        data = request.get_json()
        cycles = data.get('cycles', [])
        
        if not cycles:
            return jsonify({"error": "No cycles provided"}), 400
        
        results = []
        for cycle in cycles:
            result_json = bridge.predict_json(json.dumps(cycle))
            results.append(json.loads(result_json))
        
        return jsonify({
            "predictions": results,
            "total": len(results)
        })
        
    except Exception as e:
        return jsonify({"error": str(e), "success": False}), 400

if __name__ == '__main__':
    try:
        print("="*70)
        print("🚀 MANUFACTURING ML API SERVER")
        print("="*70)
        print("\n📡 Endpoints:")
        print("  GET  /health                    - Health check")
        print("  POST /predict                   - Single cycle prediction")
        print("  POST /predict_with_history      - Prediction with context")
        print("  POST /should_maintain           - Boolean maintenance decision")
        print("  POST /batch_predict             - Batch predictions")
        print("\n" + "="*70)
        print("🌐 Server running on: http://localhost:5001")
        print("📊 Ready to receive requests from Tecnomatix!")
        print("="*70 + "\n")
        
        app.run(host='0.0.0.0', port=5001, debug=True)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Server stopped by user")
    except Exception as e:
        print(f"\n❌ Server failed to start: {e}")
        import traceback
        traceback.print_exc()
