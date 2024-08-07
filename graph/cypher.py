class Cypher:
    """
    A class to represent and execute a Cypher command.

    Attributes:
        exec_cmd (str): The Cypher command to be executed.
    """
    def __init__(self, exec_cmd: str):
        """
        Initialize a new Cypher instance.

        Args:
            exec_cmd (str): The Cypher command to be executed.
        """       
        self.exec_cmd = exec_cmd

    def execute(self, tx, **params):
        """
        Execute the Cypher command using the provided transaction.

        Args:
            tx: The transaction object used to run the Cypher command.
            **params: Additional parameters for the Cypher command.

        Returns:
            Result: The result of the Cypher command execution.
        """
        return tx.run(self.exec_cmd, **params)
    
    def __str__(self) -> str:
        return self.exec_cmd

