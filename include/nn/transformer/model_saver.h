#ifndef MODEL_SAVER_H
#define MODEL_SAVER_H

#include <string>
#include <vector>
#include <map>
#include <fstream>
#include <nlohmann/json.hpp>
#include "../matrix.h"

using json = nlohmann::json;

/**
 * @brief Save and load models in HuggingFace-style format
 * 
 * Directory structure:
 *   model_dir/
 *     ├── config.json       (model architecture config)
 *     ├── model.bin         (model weights in binary)
 *     ├── vocab.json        (tokenizer vocabulary)
 *     └── labels.json       (intent and slot mappings)
 */
class ModelSaver {
public:
    /**
     * @brief Save model configuration to JSON
     */
    static bool saveConfig(const std::string& dir, const json& config) {
        std::ofstream file(dir + "/config.json");
        if (!file.is_open()) return false;
        file << config.dump(2);
        return true;
    }
    
    /**
     * @brief Load model configuration from JSON
     */
    static json loadConfig(const std::string& dir) {
        std::ifstream file(dir + "/config.json");
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open config.json");
        }
        json config;
        file >> config;
        return config;
    }
    
    /**
     * @brief Save Matrix to binary file
     */
    static bool saveMatrix(std::ofstream& file, const Matrix& mat) {
        size_t rows = mat.getRows();
        size_t cols = mat.getCols();
        
        file.write(reinterpret_cast<const char*>(&rows), sizeof(size_t));
        file.write(reinterpret_cast<const char*>(&cols), sizeof(size_t));
        
        for (size_t i = 0; i < rows; i++) {
            for (size_t j = 0; j < cols; j++) {
                double val = mat.get(i, j);
                file.write(reinterpret_cast<const char*>(&val), sizeof(double));
            }
        }
        return true;
    }
    
    /**
     * @brief Load Matrix from binary file
     */
    static Matrix loadMatrix(std::ifstream& file) {
        size_t rows, cols;
        file.read(reinterpret_cast<char*>(&rows), sizeof(size_t));
        file.read(reinterpret_cast<char*>(&cols), sizeof(size_t));
        
        Matrix mat(rows, cols);
        for (size_t i = 0; i < rows; i++) {
            for (size_t j = 0; j < cols; j++) {
                double val;
                file.read(reinterpret_cast<char*>(&val), sizeof(double));
                mat.set(i, j, val);
            }
        }
        return mat;
    }
    
    /**
     * @brief Save vocabulary to JSON
     */
    static bool saveVocab(const std::string& dir, 
                         const std::unordered_map<std::string, int>& vocab,
                         int pad_id, int unk_id, int bos_id, int eos_id) {
        json vocab_json;
        vocab_json["vocab"] = vocab;
        vocab_json["special_tokens"] = {
            {"pad_id", pad_id},
            {"unk_id", unk_id},
            {"bos_id", bos_id},
            {"eos_id", eos_id}
        };
        
        std::ofstream file(dir + "/vocab.json");
        if (!file.is_open()) return false;
        file << vocab_json.dump(2);
        return true;
    }
    
    /**
     * @brief Load vocabulary from JSON
     */
    static std::tuple<std::unordered_map<std::string, int>, int, int, int, int> 
    loadVocab(const std::string& dir) {
        std::ifstream file(dir + "/vocab.json");
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open vocab.json");
        }
        
        json vocab_json;
        file >> vocab_json;
        
        std::unordered_map<std::string, int> vocab = vocab_json["vocab"].get<std::unordered_map<std::string, int>>();
        int pad_id = vocab_json["special_tokens"]["pad_id"];
        int unk_id = vocab_json["special_tokens"]["unk_id"];
        int bos_id = vocab_json["special_tokens"]["bos_id"];
        int eos_id = vocab_json["special_tokens"]["eos_id"];
        
        return {vocab, pad_id, unk_id, bos_id, eos_id};
    }
    
    /**
     * @brief Save label mappings (intents and slots)
     */
    static bool saveLabels(const std::string& dir,
                          const std::unordered_map<std::string, int>& intent_to_id,
                          const std::unordered_map<int, std::string>& id_to_intent,
                          const std::unordered_map<std::string, int>& slot_to_id,
                          const std::unordered_map<int, std::string>& id_to_slot) {
        json labels_json;
        
        // Convert maps to JSON
        labels_json["intent_to_id"] = intent_to_id;
        labels_json["slot_to_id"] = slot_to_id;
        
        // Convert int keys to string for JSON
        json id_to_intent_json;
        for (const auto& [id, intent] : id_to_intent) {
            id_to_intent_json[std::to_string(id)] = intent;
        }
        labels_json["id_to_intent"] = id_to_intent_json;
        
        json id_to_slot_json;
        for (const auto& [id, slot] : id_to_slot) {
            id_to_slot_json[std::to_string(id)] = slot;
        }
        labels_json["id_to_slot"] = id_to_slot_json;
        
        std::ofstream file(dir + "/labels.json");
        if (!file.is_open()) return false;
        file << labels_json.dump(2);
        return true;
    }
    
    /**
     * @brief Load label mappings
     */
    static std::tuple<std::unordered_map<std::string, int>, std::unordered_map<int, std::string>,
                     std::unordered_map<std::string, int>, std::unordered_map<int, std::string>>
    loadLabels(const std::string& dir) {
        std::ifstream file(dir + "/labels.json");
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open labels.json");
        }
        
        json labels_json;
        file >> labels_json;
        
        std::unordered_map<std::string, int> intent_to_id = labels_json["intent_to_id"].get<std::unordered_map<std::string, int>>();
        std::unordered_map<std::string, int> slot_to_id = labels_json["slot_to_id"].get<std::unordered_map<std::string, int>>();
        
        std::unordered_map<int, std::string> id_to_intent;
        for (auto& [key, val] : labels_json["id_to_intent"].items()) {
            id_to_intent[std::stoi(key)] = val;
        }
        
        std::unordered_map<int, std::string> id_to_slot;
        for (auto& [key, val] : labels_json["id_to_slot"].items()) {
            id_to_slot[std::stoi(key)] = val;
        }
        
        return {intent_to_id, id_to_intent, slot_to_id, id_to_slot};
    }
};

#endif // MODEL_SAVER_H
