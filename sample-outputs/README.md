# 📋 Sample Outputs for Hackathon Demo

## 🎯 Overview

This folder contains example outputs that the AI system will generate during the demo. These serve as:
- **Reference implementations** for development
- **Backup materials** for demo day
- **Validation targets** for testing

## 📁 Sample Scenarios

### **Scenario 1: Basic Project Creation**

#### **Input:**
```
"Create a new S32K144 project with LQFP100 package"
```

#### **Generated Output:**
- **Maven Project**: `s32k144-project-creation/`
- **Test Class**: `ProjectCreationTest.java`
- **Execution Time**: ~2 minutes
- **Expected Result**: S32K144 project visible in S32DS Project Explorer

### **Scenario 2: Project Creation + Build**

#### **Input:**
```
"Create S32K144 project and build it"
```

#### **Generated Output:**
- **Maven Project**: `s32k144-create-and-build/`
- **Test Classes**: `ProjectCreationTest.java`, `BuildExecutionTest.java`
- **Execution Time**: ~4 minutes
- **Expected Result**: Project created and built successfully

### **Scenario 3: Custom Configuration**

#### **Input:**
```
"Create S32K144 project named 'HackathonDemo' with LQFP144 package and build with optimization level -O2"
```

#### **Generated Output:**
- **Maven Project**: `s32k144-custom-config/`
- **Test Classes**: Multiple test classes with custom parameters
- **Execution Time**: ~5 minutes
- **Expected Result**: Customized project with specific build settings

## 🧪 Sample Test Code

### **ProjectCreationTest.java (Data-Driven)**
```java
package com.hackathon.demo;

import static org.junit.Assert.*;
import org.eclipse.swtbot.eclipse.finder.SWTWorkbenchBot;
import org.eclipse.swtbot.swt.finder.junit.SWTBotJunit4ClassRunner;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;
import java.io.File;
import java.util.Arrays;
import java.util.Collection;

@RunWith(Parameterized.class)
public class ProjectCreationTest {

    private SWTWorkbenchBot bot;
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
    
    @Before
    public void setUp() {
        bot = new SWTWorkbenchBot();
        bot.perspectiveByLabel("S32DS").activate();
    }
    
    @Test
    public void testCreateProject() {
        // Step 1: Open New Project Wizard (data-driven)
        openNewProjectWizard(projectFileName, socType);

        // Step 2: Select SoC Configuration (data-driven)
        selectSoCConfiguration(socType, packageType, sdkType);

        // Step 3: Configure Project Settings (data-driven)
        configureProjectSettings(projectName, packageType, projectFileName, socType);

        // Step 4: Finish Project Creation (data-driven)
        finishProjectCreation(projectFileName, socType);

        // Step 5: Verify Project Created (data-driven)
        verifyProjectCreated(projectName, projectFileName, socType);
    }
    
    private void openNewProjectWizard() {
        bot.menu("File").menu("New").menu("S32DS Project").click();
        bot.waitUntil(shellIsActive("New S32DS Project"));
        assertTrue("New S32DS Project wizard should be active",
                   bot.shell("New S32DS Project").isActive());
    }
    
    private void selectS32K144Configuration() {
        bot.tree().select("S32K1 SDK");
        bot.button("Next >").click();
        bot.comboBoxWithLabel("Processor:").setSelection(SOC_TYPE);
        assertEquals("Processor should be " + SOC_TYPE, SOC_TYPE,
                     bot.comboBoxWithLabel("Processor:").selection());
    }
    
    private void configureProjectSettings(String projectName, String packageType) {
        bot.textWithLabel("Project name:").setText(projectName);
        bot.comboBoxWithLabel("Package:").setSelection(packageType);
        
        assertEquals("Project name should be set", projectName,
                     bot.textWithLabel("Project name:").getText());
        assertEquals("Package should be " + packageType, packageType,
                     bot.comboBoxWithLabel("Package:").selection());
    }
    
    private void finishProjectCreation() {
        bot.button("Finish").click();
        bot.waitUntil(shellCloses(bot.shell("New S32DS Project")));
    }
    
    private void verifyProjectCreated(String projectName) {
        assertTrue("Project should be visible in explorer",
                   bot.tree().hasItems(projectName));
    }
    
    @After
    public void tearDown() {
        // Clean up if needed
    }
}
```

### **BuildExecutionTest.java (Data-Driven)**
```java
package com.hackathon.demo;

import static org.junit.Assert.*;
import org.eclipse.swtbot.eclipse.finder.SWTWorkbenchBot;
import org.eclipse.swtbot.swt.finder.junit.SWTBotJunit4ClassRunner;
import org.eclipse.swtbot.swt.finder.waits.DefaultCondition;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;
import java.util.Arrays;
import java.util.Collection;

@RunWith(Parameterized.class)
public class BuildExecutionTest {

    private SWTWorkbenchBot bot;
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
    
    @Before
    public void setUp() {
        bot = new SWTWorkbenchBot();
        bot.perspectiveByLabel("S32DS").activate();
    }
    
    @Test
    public void testBuildProject() {
        // Step 1: Select Project (data-driven)
        selectProject(projectName, projectFileName, socType);

        // Step 2: Execute Build (data-driven)
        executeBuild(buildConfig, socType);

        // Step 3: Verify Build Success (data-driven)
        verifyBuildSuccess(projectFileName, socType, expectedBuildTime);
    }
    
    private void selectProject(String projectName) {
        bot.tree().select(projectName);
        assertTrue("Project should be selected",
                   bot.tree().isSelected(projectName));
    }
    
    private void executeBuild() {
        bot.menu("Project").menu("Build Project").click();
        
        // Wait for build to complete
        bot.waitUntil(new DefaultCondition() {
            public boolean test() throws Exception {
                String consoleText = bot.viewByTitle("Console").bot()
                    .styledText().getText();
                return consoleText.contains("Build Finished") ||
                       consoleText.contains("Build completed");
            }
            
            public String getFailureMessage() {
                return "Build did not complete within timeout";
            }
        }, 60000); // 60 second timeout
    }
    
    private void verifyBuildSuccess() {
        // Check console for success message
        String consoleText = bot.viewByTitle("Console").bot()
            .styledText().getText();
        assertTrue("Build should succeed", 
                   consoleText.contains("Build Finished") ||
                   consoleText.contains("Build completed"));
        
        // Check no errors in Problems view
        assertFalse("No build errors should exist",
                    bot.viewByTitle("Problems").bot().tree()
                        .hasItems("Errors"));
    }
}
```

## 📦 Maven Project Structure

### **Generated pom.xml**
```xml
<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 
         http://maven.apache.org/xsd/maven-4.0.0.xsd">
    <modelVersion>4.0.0</modelVersion>
    
    <groupId>com.hackathon.demo</groupId>
    <artifactId>s32k144-swtbot-tests</artifactId>
    <version>1.0.0</version>
    <name>S32K144 SWTBot Demo Tests</name>
    <description>Generated SWTBot tests for S32K144 project creation and build</description>
    
    <properties>
        <maven.compiler.source>11</maven.compiler.source>
        <maven.compiler.target>11</maven.compiler.target>
        <project.build.sourceEncoding>UTF-8</project.build.sourceEncoding>
        <swtbot.version>3.1.0</swtbot.version>
        <junit.version>4.13.2</junit.version>
    </properties>
    
    <dependencies>
        <!-- SWTBot Dependencies -->
        <dependency>
            <groupId>org.eclipse.swtbot</groupId>
            <artifactId>org.eclipse.swtbot.eclipse.finder</artifactId>
            <version>${swtbot.version}</version>
            <scope>test</scope>
        </dependency>
        
        <dependency>
            <groupId>org.eclipse.swtbot</groupId>
            <artifactId>org.eclipse.swtbot.junit4_x</artifactId>
            <version>${swtbot.version}</version>
            <scope>test</scope>
        </dependency>
        
        <!-- JUnit -->
        <dependency>
            <groupId>junit</groupId>
            <artifactId>junit</artifactId>
            <version>${junit.version}</version>
            <scope>test</scope>
        </dependency>
    </dependencies>
    
    <build>
        <plugins>
            <plugin>
                <groupId>org.apache.maven.plugins</groupId>
                <artifactId>maven-compiler-plugin</artifactId>
                <version>3.8.1</version>
                <configuration>
                    <source>11</source>
                    <target>11</target>
                </configuration>
            </plugin>
            
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
                        <property>
                            <name>org.eclipse.swtbot.playback.delay</name>
                            <value>100</value>
                        </property>
                    </systemProperties>
                    <includes>
                        <include>**/*Test.java</include>
                    </includes>
                </configuration>
            </plugin>
        </plugins>
    </build>
</project>
```

## 🎯 Demo Execution Commands

### **Generate and Run Tests**
```bash
# Generate tests from user input
python demo_generator.py --input "Create S32K144 project and build it"

# Navigate to generated project
cd generated-s32k144-tests/

# Execute tests
mvn clean test

# View results
cat target/surefire-reports/TEST-*.xml
```

### **Expected Output**
```
[INFO] Running com.hackathon.demo.ProjectCreationTest
[INFO] Tests run: 1, Failures: 0, Errors: 0, Skipped: 0
[INFO] Running com.hackathon.demo.BuildExecutionTest  
[INFO] Tests run: 1, Failures: 0, Errors: 0, Skipped: 0
[INFO] BUILD SUCCESS
```

## 📊 Performance Metrics

### **Timing Expectations**
- **Code Generation**: 5-10 seconds
- **Project Creation Test**: 90-120 seconds
- **Build Execution Test**: 120-180 seconds
- **Total Demo Time**: 5-7 minutes

### **Success Criteria**
- [ ] All tests pass without errors
- [ ] S32K144 project visible in S32DS
- [ ] Build completes successfully
- [ ] No errors in Problems view
- [ ] Console shows "Build Finished"

## 🎯 Data-Driven Testing Benefits

### **Scalability**
- **Easy SoC Addition**: Add new SoCs by updating test data only
- **No Code Changes**: Test logic remains generic and reusable
- **Configuration Management**: Centralized SoC specifications

### **Maintainability**
- **Single Source of Truth**: All SoC data in configuration files
- **Generic Test Methods**: No SoC-specific hardcoded logic
- **Parameterized Execution**: JUnit handles test variations automatically

### **Demo Advantages**
- **Multiple SoCs**: Demonstrate support for S32K144, S32K146, S32G274A
- **Easy Customization**: Change test data without code modification
- **Professional Approach**: Industry-standard data-driven testing pattern

### **AI Generation Benefits**
- **Template Reuse**: Same code template works for all SoCs
- **Data Focus**: AI generates test data, not SoC-specific code
- **Flexible Output**: Easy to add new SoCs to generated tests

Ready for demo day! 🎉
