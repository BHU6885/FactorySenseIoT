CREATE DATABASE IF NOT EXISTS factorysenseiot
    CHARACTER SET utf8mb4
    COLLATE utf8mb4_unicode_ci;

USE factorysenseiot;

CREATE TABLE IF NOT EXISTS prediction_history (
    prediction_id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    machine_id VARCHAR(100) NULL,
    prediction_type ENUM('failure', 'rul') NOT NULL,
    input_data JSON NOT NULL,
    predicted_result JSON NOT NULL,
    failure_probability DECIMAL(5, 2) NULL,
    predicted_rul DECIMAL(12, 2) NULL,
    risk_level VARCHAR(20) NULL,
    model_name VARCHAR(100) NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (prediction_id),
    INDEX idx_prediction_history_created_at (created_at),
    INDEX idx_prediction_history_type_created (prediction_type, created_at),
    INDEX idx_prediction_history_machine (machine_id)
) ENGINE=InnoDB
  DEFAULT CHARACTER SET utf8mb4
  COLLATE utf8mb4_unicode_ci;
