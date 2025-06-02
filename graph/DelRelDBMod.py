from my_dataclasses import Relationship as Rel
from typing import List
import sqlite3

class DelRelDBM:
    """
    This class is used to store Delited Relationships in a SQLite database.
    """

    def __init__(self, db_name="deleted_relationships.db"):
        self.db_name = db_name
        self.conn = sqlite3.connect(self.db_name, check_same_thread=False) # Allow multiple threads to access the database
        self.cursor = self.conn.cursor()
        self.__create_table()

    def __create_table(self):
        """Create the table for deleted relationships if it doesn't exist."""
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS deleted_relationships (
                source TEXT,
                target TEXT,
                type TEXT
            )
        ''')
        self.conn.commit()

    def add_deleted_relationship(self, relationship: Rel):
        """
        Add a deleted relationship to the database.
        :param relationship: Relationship object to be added.
        """
        self.cursor.execute('''
            INSERT INTO deleted_relationships (source, target, type)
            VALUES (?, ?, ?)
        ''', (relationship.source, relationship.target, relationship.type))
        self.conn.commit()
        return True
    
    def remove_deleted_relationship(self, relationship: Rel):
        """
        Remove a deleted relationship from the database.
        :param relationship: Relationship object to be removed.
        """
        self.cursor.execute('''
            DELETE FROM deleted_relationships
            WHERE source = ? AND target = ? AND type = ?
        ''', (relationship.source, relationship.target, relationship.type))
        self.conn.commit()
        return True

    def get_deleted_relationships(self) -> List[Rel]:
        """
        Retrieve all deleted relationships from the database.
        :return: List of Relationship objects.
        """
        self.cursor.execute('SELECT source, target, type FROM deleted_relationships')
        rows = self.cursor.fetchall()
        return [Rel(source=row[0], target=row[1], type=row[2]) for row in rows]

    def has_deleted_relationship(self, relationship: Rel) -> bool:
        """
        Check if a specific deleted relationship exists in the database.
        :param relationship: Relationship object to check.
        :return: True if the relationship exists, False otherwise.
        """
        self.cursor.execute('''
            SELECT 1 FROM deleted_relationships
            WHERE source = ? AND target = ? AND type = ?
        ''', (relationship.source, relationship.target, relationship.type))
        return self.cursor.fetchone() is not None
    
    def clear_deleted_relationships(self):
        """Clear all deleted relationships from the database."""
        self.cursor.execute('DELETE FROM deleted_relationships')
        self.conn.commit()
        return True

    def close(self):
        """Close the database connection."""
        self.conn.close()