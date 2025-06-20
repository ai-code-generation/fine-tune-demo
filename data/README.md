# S32K144 Training Data for Fine-Tuning

This directory contains comprehensive training data for fine-tuning language models to assist with S32K144 microcontroller project automation using SWTBot in Eclipse IDE.

## 📁 Data Files

### `train.yaml` - Training Data (Primary Dataset)
- **Examples**: 8 comprehensive training examples
- **Focus**: Core S32K144 project operations and configurations
- **Content**:
  - Project creation and build configuration
  - GPIO driver integration
  - Debugger setup with PEMicro interface
  - FlexCAN automotive communication
  - ADC sensor reading configuration
  - PWM motor control setup
  - Project import and automotive configuration
  - Memory Protection Unit (MPU) safety configuration

### `val.yaml` - Validation Data
- **Examples**: 4 validation examples
- **Focus**: Communication protocols and system configuration
- **Content**:
  - UART communication setup
  - External interrupt handling
  - System clock optimization
  - SPI sensor interface configuration

### `test.yaml` - Test Data
- **Examples**: 4 test examples
- **Focus**: Safety features and advanced peripherals
- **Content**:
  - Watchdog timer configuration
  - Low power mode management
  - I2C sensor networks
  - Timer interrupt configuration

## 🎯 Training Data Characteristics

### Total Dataset Size
- **Training Examples**: 8 (primary learning)
- **Validation Examples**: 4 (model validation)
- **Test Examples**: 4 (final evaluation)
- **Total**: 16 comprehensive examples

### Coverage Areas

#### 1. **Project Management**
- Project creation and configuration
- Build settings optimization
- Import existing projects
- Automotive-specific configurations

#### 2. **Peripheral Configuration**
- GPIO (General Purpose I/O)
- ADC (Analog-to-Digital Converter)
- PWM (Pulse Width Modulation)
- FlexCAN (Controller Area Network)
- UART/LPUART (Serial Communication)
- SPI/LPSPI (Serial Peripheral Interface)
- I2C/LPI2C (Inter-Integrated Circuit)

#### 3. **Safety & Security**
- Memory Protection Unit (MPU)
- Watchdog timer configuration
- Interrupt handling
- Fault protection mechanisms

#### 4. **System Features**
- Clock configuration and optimization
- Power management and low power modes
- Timer and interrupt systems
- Debug interface setup

#### 5. **Automotive Applications**
- CAN bus communication
- Motor control with PWM
- Sensor interfacing
- Safety-critical configurations

## 🔧 Data Format

Each training example follows the conversation format:

```yaml
---
messages:
  - role: system
    content: "You are an expert test automation assistant specializing in S32K144 microcontroller projects using SWTBot automation in Eclipse IDE."
  - role: user
    content: "Task description or question about S32K144 configuration"
  - role: assistant
    content: |
      ```java
      @Test
      public void testMethodName() {
          // Complete SWTBot automation code
          // for the requested S32K144 task
      }
      ```
```

## 🎯 Training Objectives

The training data is designed to teach the model to:

1. **Generate SWTBot Test Code**: Create complete, executable SWTBot test methods
2. **S32K144 Expertise**: Understand S32K144 microcontroller capabilities and configurations
3. **Eclipse IDE Navigation**: Navigate Eclipse IDE menus, dialogs, and project structures
4. **Automotive Standards**: Apply automotive development best practices
5. **Safety Considerations**: Implement safety-critical configurations
6. **Code Integration**: Generate code that integrates with S32K SDK

## 🚀 Usage with Fine-Tuning Pipeline

Use these files with the fine-tuning pipeline:

```bash
python scripts/run_pipeline.py \
  --model-type llama2 \
  --model-size 7b \
  --train-file data/train.yaml \
  --val-file data/val.yaml \
  --test-file data/test.yaml \
  --max-epochs 3 \
  --gpus 1
```

## 📊 Expected Training Results

After fine-tuning with this dataset, the model should be able to:

- Generate complete SWTBot test automation code for S32K144 projects
- Configure various S32K144 peripherals through Eclipse IDE automation
- Apply automotive development best practices
- Handle safety-critical configurations
- Integrate with S32K SDK and NXP tools
- Navigate Eclipse IDE efficiently through automation

## 🔍 Data Quality Features

- **Realistic Scenarios**: Based on actual S32K144 development workflows
- **Complete Code**: Full SWTBot test methods with proper error handling
- **Best Practices**: Follows automotive and embedded development standards
- **Comprehensive Coverage**: Covers major S32K144 peripherals and features
- **Consistent Format**: Uniform structure across all examples
- **Practical Applications**: Real-world automotive use cases

## 📝 Notes

- All examples use the project name "S32KDUC1431_Project" for consistency
- Code includes proper wait conditions and error handling
- Examples cover both basic and advanced S32K144 features
- Automotive safety considerations are integrated throughout
- Compatible with S32 Design Studio and Eclipse-based IDEs

This training data provides a solid foundation for creating an AI assistant specialized in S32K144 microcontroller project automation.
