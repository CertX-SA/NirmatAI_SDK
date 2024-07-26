pipeline {
    agent any

    stages {

        stage("Checkout") {
            steps {
                checkout scm
                script {
                    // Set the branch name as an environment variable
                    env.BRANCH_NAME = sh(script: "git rev-parse --abbrev-ref HEAD", returnStdout: true).trim()
                }
                echo "Branch name: ${env.BRANCH_NAME}"
            }
        }

        stage("DCV Pull") {
            steps {
                sh "dvc pull"
            }
        }

        stage("Run Experiments") {
            when {
                expression {
                    return env.BRANCH_NAME.contains("experiment")
                }
            }
            steps {
                lock(resource: "pipeline-lock") {
                    withEnv(["POD_NAME=${env.BRANCH_NAME}"]) {
                        // Run experiments
                        sh "make run-experiments"
                        // Stop Ansible playbook
                        sh "make stop"
                    }
                }
            }
        }

        stage("Deploy") {
            when {
                expression {
                    return env.BRANCH_NAME.contains("HEAD")
                }
            }
            steps {
                echo "Deploying to production"
                // Stop previous pod
                sh "JENKINS_NODE_COOKIE=dontKillMe ansible-playbook deploy.yml"
            }
        }
    }
    triggers {pollSCM("* * * * *")}
}
