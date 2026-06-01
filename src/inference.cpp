#include <iostream>
#include <vector>
#include <iomanip>
#include <LightGBM/c_api.h>

int main() {
    std::cout << "\n=== Quant C++ Inference Engine ===" << std::endl;

    // 1. load model
    BoosterHandle handle;
    int out_num_iterations;
    
    // path to model.txt
    int result = BoosterCreateFromModelfile("models/lgbm_model.txt", &out_num_iterations, &handle);

    if (result != 0) {
        std::cerr << "[Error] Could not load model. Does models/lgbm_model.txt exist?" << std::endl;
        return -1;
    }
    std::cout << "[OK] LightGBM model successfully loaded in C++." << std::endl;

    // 2. define sample features (log_return, volatility_20d, z_score_20d)
    // assume the market had a small positive return today, 
    // moderate volatility, and a high momentum z-score
    std::vector<double> features = {0.0015, 0.0120, 1.500};

    // 3. run inference
    const void* in_p = features.data();
    double out_result[1];
    int64_t out_len;

    // call LGBM C-API
    result = BoosterPredictForMat(
        handle,
        in_p,
        C_API_DTYPE_FLOAT64,
        1,       // number of rows (we have 1 sample)
        3,       // number of columns (we have 3 features)
        1,       // is_row_major (1 = true)
        C_API_PREDICT_NORMAL,
        0,       // start_iteration
        -1,      // num_iteration (-1 means use all trees)
        "",      // parameters
        &out_len,
        out_result
    );

    if (result != 0) {
        std::cerr << "[Error] Prediction failed!" << std::endl;
        BoosterFree(handle);
        return -1;
    }

    // 4. output results
    std::cout << "\n--- Signal Generation ---" << std::endl;
    std::cout << "Input Features: Return=" << features[0] 
              << ", Volatility=" << features[1] 
              << ", Z-Score=" << features[2] << std::endl;
              
    std::cout << "Predicted Probability (Next Day UP): " 
              << std::fixed << std::setprecision(4) << out_result[0] << std::endl;

    if (out_result[0] > 0.5) {
        std::cout << "--> Trading Signal: BUY (1)" << std::endl;
    } else {
        std::cout << "--> Trading Signal: SELL/HOLD (0)" << std::endl;
    }
    std::cout << "==================================\n" << std::endl;

    // free memory safely
    BoosterFree(handle);
    return 0;
}