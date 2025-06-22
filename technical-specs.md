# 🔧 Technical Specifications for Hackathon Demo

## 🏗️ System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   User Input    │───▶│   AI Generator  │───▶│  Maven Project  │
│  (Text/CLI)     │    │  (RAG + LLM)    │    │  (SWTBot Tests) │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │ Knowledge Base  │    │   S32DS IDE     │
                       │ (S32K144 Data)  │    │ (Live Testing)  │
                       └─────────────────┘    └─────────────────┘
```

## 🧠 AI Components

### **1. Simple RAG System**
```python
# Minimal RAG implementation
class S32K144RAG:
    def __init__(self):
        self.patterns = {
            "create_project": {
                "keywords": ["create", "new", "project", "s32k144"],
                "template": "project_creation_template.java",
                "steps": ["open_wizard", "select_soc", "configure", "finish"]
            },
            "build_project": {
                "keywords": ["build", "compile", "make"],
                "template": "build_execution_template.java", 
                "steps": ["select_project", "build", "verify"]
            }
        }
    
    def search(self, query):
        # Simple keyword matching
        for pattern_name, pattern in self.patterns.items():
            if any(keyword in query.lower() for keyword in pattern["keywords"]):
                return pattern
        return None
```

### **2. Code Generation Pipeline**
```python
class CodeGenerator:
    def generate_swtbot_test(self, user_input):
        # 1. Parse input
        intent = self.parse_intent(user_input)
        
        # 2. Retrieve relevant patterns
        pattern = self.rag.search(user_input)
        
        # 3. Generate code from template
        code = self.fill_template(pattern, intent)
        
        # 4. Create Maven project
        maven_project = self.create_maven_project(code)
        
        return maven_project
```

## 🧪 SWTBot Test Templates

### **Template 1: Data-Driven Project Creation**
```java
// File: ProjectCreationTest.java
@RunWith(Parameterized.class)
public class ProjectCreationTest {

    private String socType;
    private String packageType;
    private String projectName;
    private String projectFileName;
    private String sdkType;

    public ProjectCreationTest(String socType, String packageType, String projectName,
                              String projectFileName, String sdkType) {
        this.socType = socType;
        this.packageType = packageType;
        this.projectName = projectName;
        this.projectFileName = projectFileName;
        this.sdkType = sdkType;
    }

    @Parameters(name = "SoC: {0}, Package: {1}")
    public static Collection<Object[]> testData() {
        return Arrays.asList(new Object[][] {
            {"S32K144", "LQFP100", "HackathonDemo_S32K144", "s32k144_demo.s32ds", "S32K1 SDK"},
            {"S32K146", "LQFP144", "HackathonDemo_S32K146", "s32k146_demo.s32ds", "S32K1 SDK"},
            {"S32G274A", "BGA", "HackathonDemo_S32G274A", "s32g274a_demo.s32ds", "S32G SDK"}
        });
    }

    @Test
    public void testCreateProject() {
        // Step 1: Open New Project Wizard
        openNewProjectWizard(projectFileName, socType);

        // Step 2: Select SoC Configuration
        selectSoCConfiguration(socType, packageType, sdkType);

        // Step 3: Configure Project Settings
        configureProjectSettings(projectName, packageType, projectFileName, socType);

        // Step 4: Finish Project Creation
        finishProjectCreation(projectFileName, socType);

        // Step 5: Verify Project Created
        verifyProjectCreated(projectName, projectFileName, socType);
    }

private void openNewProjectWizard(String projectFileName, String socType) {
    bot.menu("File").menu("New").menu("S32DS Project").click();
    bot.waitUntil(shellIsActive("New S32DS Project"));
    // Log project file name and SoC type for tracking
    System.out.println("Creating " + socType + " project with file: " + projectFileName);
}

private void selectSoCConfiguration(String socType, String packageType, String sdkType) {
    // Select SDK based on test data
    bot.tree().select(sdkType);
    bot.button("Next >").click();
    bot.comboBoxWithLabel("Processor:").setSelection(socType);
    assertEquals(socType + " processor should be selected", socType,
                 bot.comboBoxWithLabel("Processor:").selection());
}

private void configureProjectSettings(String projectName, String packageType, String projectFileName, String socType) {
    bot.textWithLabel("Project name:").setText(projectName);
    bot.comboBoxWithLabel("Package:").setSelection(packageType);
    // Set project file location if needed
    bot.textWithLabel("Location:").setText("C:/S32DS_Workspace/" + projectFileName);
    System.out.println("Configured " + socType + " project: " + projectName);
}

private void finishProjectCreation(String projectFileName, String socType) {
    bot.button("Finish").click();
    bot.waitUntil(shellCloses(bot.shell("New S32DS Project")));
    System.out.println(socType + " project creation completed for: " + projectFileName);
}

private void verifyProjectCreated(String projectName, String projectFileName, String socType) {
    assertTrue(socType + " project should be visible in explorer",
               bot.tree().hasItems(projectName));
    // Verify project file exists
    assertTrue("Project file should exist: " + projectFileName,
               new File("C:/S32DS_Workspace/" + projectFileName).exists());
}

// Note: SDK mapping is now handled by test data parameters
// No hardcoded SoC-specific logic needed in the test methods
```

### **Template 2: Data-Driven Build Execution**
```java
// File: BuildExecutionTest.java
@RunWith(Parameterized.class)
public class BuildExecutionTest {

    private String socType;
    private String projectName;
    private String projectFileName;
    private String buildConfig;
    private String expectedBuildTime;

    public BuildExecutionTest(String socType, String projectName, String projectFileName,
                             String buildConfig, String expectedBuildTime) {
        this.socType = socType;
        this.projectName = projectName;
        this.projectFileName = projectFileName;
        this.buildConfig = buildConfig;
        this.expectedBuildTime = expectedBuildTime;
    }

    @Parameters(name = "SoC: {0}, Config: {3}")
    public static Collection<Object[]> testData() {
        return Arrays.asList(new Object[][] {
            {"S32K144", "HackathonDemo_S32K144", "s32k144_demo.s32ds", "Debug", "120"},
            {"S32K146", "HackathonDemo_S32K146", "s32k146_demo.s32ds", "Release", "90"},
            {"S32G274A", "HackathonDemo_S32G274A", "s32g274a_demo.s32ds", "Debug", "180"}
        });
    }

    @Test
    public void testBuildProject() {
        // Step 1: Select Project
        selectProject(projectName, projectFileName, socType);

        // Step 2: Execute Build
        executeBuild(buildConfig, socType);

        // Step 3: Verify Build Success
        verifyBuildSuccess(projectFileName, socType, expectedBuildTime);
    }

private void selectProject(String projectName) {
    bot.tree().select(projectName);
}

private void executeBuild() {
    bot.menu("Project").menu("Build Project").click();
    
    // Wait for build to complete
    bot.waitUntil(new DefaultCondition() {
        public boolean test() throws Exception {
            return bot.viewByTitle("Console").bot().styledText()
                .getText().contains("Build Finished");
        }
        public String getFailureMessage() {
            return "Build did not complete";
        }
    }, 60000); // 60 second timeout
}

private void verifyBuildSuccess() {
    // Check console for success message
    String consoleText = bot.viewByTitle("Console").bot().styledText().getText();
    assertTrue("Build should succeed", consoleText.contains("Build Finished"));
    
    // Check no errors in Problems view
    assertFalse("No build errors should exist",
                bot.viewByTitle("Problems").bot().tree().hasItems("Errors"));
}
```

## 📦 Maven Project Structure

```
generated-soc-demo/
├── pom.xml                          # Maven configuration
├── src/
│   └── test/java/
│       └── com/hackathon/demo/
│           ├── ProjectCreationTest.java     # Data-driven project creation
│           ├── BuildExecutionTest.java      # Data-driven build execution
│           ├── TestBase.java               # Common setup
│           └── data/
│               ├── SoCTestData.java        # Test data provider
│               └── TestDataLoader.java     # Data loading utilities
├── test-data/
│   ├── soc-configs/
│   │   ├── s32k144-config.json
│   │   ├── s32k146-config.json
│   │   └── s32g274a-config.json
│   ├── expected-results/
│   └── test-parameters.csv              # CSV-driven test data
└── README.md                            # Generated instructions
```

## 📊 Test Data Configuration

### **SoC Test Data Provider**
```java
// File: SoCTestData.java
public class SoCTestData {

    public static Collection<Object[]> getProjectCreationData() {
        return Arrays.asList(new Object[][] {
            // {socType, packageType, projectName, projectFileName, sdkType}
            {"S32K144", "LQFP100", "HackathonDemo_S32K144", "s32k144_demo.s32ds", "S32K1 SDK"},
            {"S32K146", "LQFP144", "HackathonDemo_S32K146", "s32k146_demo.s32ds", "S32K1 SDK"},
            {"S32K148", "MAPBGA100", "HackathonDemo_S32K148", "s32k148_demo.s32ds", "S32K1 SDK"},
            {"S32G274A", "BGA", "HackathonDemo_S32G274A", "s32g274a_demo.s32ds", "S32G SDK"},
            {"S32G399A", "FCBGA", "HackathonDemo_S32G399A", "s32g399a_demo.s32ds", "S32G SDK"}
        });
    }

    public static Collection<Object[]> getBuildExecutionData() {
        return Arrays.asList(new Object[][] {
            // {socType, projectName, projectFileName, buildConfig, expectedBuildTime}
            {"S32K144", "HackathonDemo_S32K144", "s32k144_demo.s32ds", "Debug", "120"},
            {"S32K144", "HackathonDemo_S32K144", "s32k144_demo.s32ds", "Release", "90"},
            {"S32K146", "HackathonDemo_S32K146", "s32k146_demo.s32ds", "Debug", "130"},
            {"S32G274A", "HackathonDemo_S32G274A", "s32g274a_demo.s32ds", "Debug", "180"},
            {"S32G274A", "HackathonDemo_S32G274A", "s32g274a_demo.s32ds", "Release", "150"}
        });
    }
}
```

### **CSV-Based Test Data**
```csv
# File: test-parameters.csv
test_type,soc_type,package_type,project_name,project_file_name,sdk_type,build_config,expected_build_time
project_creation,S32K144,LQFP100,HackathonDemo_S32K144,s32k144_demo.s32ds,S32K1 SDK,,
project_creation,S32K146,LQFP144,HackathonDemo_S32K146,s32k146_demo.s32ds,S32K1 SDK,,
project_creation,S32G274A,BGA,HackathonDemo_S32G274A,s32g274a_demo.s32ds,S32G SDK,,
build_execution,S32K144,LQFP100,HackathonDemo_S32K144,s32k144_demo.s32ds,S32K1 SDK,Debug,120
build_execution,S32K144,LQFP100,HackathonDemo_S32K144,s32k144_demo.s32ds,S32K1 SDK,Release,90
build_execution,S32G274A,BGA,HackathonDemo_S32G274A,s32g274a_demo.s32ds,S32G SDK,Debug,180
```

### **JSON Configuration Files**
```json
// File: s32k144-config.json
{
  "socType": "S32K144",
  "family": "S32K1",
  "core": "ARM Cortex-M4F",
  "frequency": "112MHz",
  "flash": "1MB",
  "ram": "128KB",
  "packages": ["LQFP100", "LQFP144", "MAPBGA100"],
  "sdkType": "S32K1 SDK",
  "buildConfigs": ["Debug", "Release"],
  "expectedBuildTimes": {
    "Debug": 120,
    "Release": 90
  },
  "peripherals": ["GPIO", "FlexCAN", "UART", "SPI", "I2C", "ADC"]
}
```

### **Maven POM Template**
```xml
<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0">
    <modelVersion>4.0.0</modelVersion>
    
    <groupId>com.hackathon.demo</groupId>
    <artifactId>s32k144-swtbot-tests</artifactId>
    <version>1.0.0</version>
    
    <properties>
        <maven.compiler.source>11</maven.compiler.source>
        <maven.compiler.target>11</maven.compiler.target>
        <swtbot.version>3.1.0</swtbot.version>
        <junit.version>4.13.2</junit.version>
    </properties>
    
    <dependencies>
        <!-- SWTBot Dependencies -->
        <dependency>
            <groupId>org.eclipse.swtbot</groupId>
            <artifactId>org.eclipse.swtbot.eclipse.finder</artifactId>
            <version>${swtbot.version}</version>
        </dependency>
        
        <!-- JUnit -->
        <dependency>
            <groupId>junit</groupId>
            <artifactId>junit</artifactId>
            <version>${junit.version}</version>
        </dependency>
    </dependencies>
    
    <build>
        <plugins>
            <plugin>
                <groupId>org.apache.maven.plugins</groupId>
                <artifactId>maven-surefire-plugin</artifactId>
                <version>3.0.0-M7</version>
                <configuration>
                    <systemProperties>
                        <property>
                            <name>org.eclipse.swtbot.search.timeout</name>
                            <value>30000</value>
                        </property>
                    </systemProperties>
                </configuration>
            </plugin>
        </plugins>
    </build>
</project>
```

## 🖥️ Demo Interface

### **Command Line Interface**
```bash
# Basic usage
./generate-demo.sh "Create S32K144 project and build it"

# With parameters
./generate-demo.sh --input "project_request.txt" --output "generated-tests/"

# Interactive mode
./generate-demo.sh --interactive
```

### **Input Format**
```
# Simple text input
Create a new S32K144 project with LQFP100 package and build it

# Structured input (optional)
{
  "action": "create_and_build",
  "soc": "S32K144", 
  "package": "LQFP100",
  "project_name": "DemoProject"
}
```

### **Output Format**
```
Generated Maven project: ./output/s32k144-demo-tests/
├── Complete SWTBot test suite
├── Ready to execute with: mvn test
├── Estimated execution time: 2-3 minutes
└── Expected results: Project created and built successfully
```

## 🔧 Environment Requirements

### **Development Environment**
- **OS**: Windows 10/11 (for S32DS compatibility)
- **Java**: JDK 11 or higher
- **Maven**: 3.6+ 
- **S32DS**: Latest version with S32K144 SDK
- **Python**: 3.8+ (for AI components)

### **AI Environment**
- **FPT AI Factory**: Primary training infrastructure
- **Nvidia NIM**: Inference microservices (if available)
- **Nvidia CUDA/TensorRT**: GPU acceleration for inference
- **Ollama**: Local LLM runtime (backup)
- **Model**: CodeLlama-7B or Nvidia optimized models
- **Vector DB**: Simple in-memory store with Nvidia Rapids acceleration
- **Dependencies**: transformers, sentence-transformers, nvidia-ml-py

### **Demo Environment**
- **Display**: Dual monitor setup for live demo
- **S32DS**: Pre-configured with S32K144 SDK
- **Backup**: VM snapshots for quick recovery

## 📊 Performance Targets

### **Code Generation**
- **Response Time**: < 10 seconds for simple requests
- **Accuracy**: 90%+ for basic project creation/build
- **Reliability**: 95%+ success rate in controlled environment

### **SWTBot Execution**
- **Project Creation**: < 2 minutes
- **Build Execution**: < 3 minutes  
- **Total Demo Time**: < 7 minutes end-to-end

## 🚀 Deployment Strategy

### **Demo Day Setup**
1. **Pre-configured Environment**: S32DS ready with clean workspace
2. **Generated Samples**: Pre-generated code for backup
3. **Multiple Scenarios**: 3-4 different demo scenarios ready
4. **Fallback Options**: Pre-recorded videos if live demo fails

### **Technical Validation**
- [ ] End-to-end testing completed
- [ ] Multiple demo runs successful
- [ ] Error scenarios handled gracefully
- [ ] Performance meets targets

Ready for implementation! 🎯
