package experiments;

import ch.qos.logback.classic.Level;
import ch.qos.logback.classic.LoggerContext;
import ch.qos.logback.classic.encoder.PatternLayoutEncoder;
import ch.qos.logback.core.ConsoleAppender;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class MainExperimentLog {

    public static void logInfo() {
        // Get the Logger context
        LoggerContext loggerContext = (LoggerContext) LoggerFactory.getILoggerFactory();

        // Create a Console Appender
        ConsoleAppender consoleAppender = new ConsoleAppender();
        consoleAppender.setContext(loggerContext);

        // Set the output pattern
        PatternLayoutEncoder encoder = new PatternLayoutEncoder();
        encoder.setContext(loggerContext);
        encoder.setPattern("%d{yyyy-MM-dd HH:mm:ss} %-5level [%thread] %logger{36} - %msg%n");
        encoder.start();

        consoleAppender.setEncoder(encoder);
        consoleAppender.start();

        // Set the root logger level to INFO and add the console appender
        ch.qos.logback.classic.Logger rootLogger = loggerContext.getLogger(Logger.ROOT_LOGGER_NAME);
        rootLogger.setLevel(Level.INFO);
        rootLogger.addAppender(consoleAppender);

        // Set the log level for DL4J to INFO
        ch.qos.logback.classic.Logger dl4jLogger = loggerContext.getLogger("org.deeplearning4j");
        dl4jLogger.setLevel(Level.INFO);
    }

    public static void main(String[] args){
        logInfo();
        System.setProperty("org.bytedeco.openblas.load", "mkl");
        new NNExperiment().evaluate();
    }
}
