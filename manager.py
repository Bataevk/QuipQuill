class Manager:
    '''Main agent who controls the system'''
    def __init__(self):
        self.database = None
    
    def _add_locations(self, locations):
        # self.database.add_locations(locations)
        return None
    
    def _add_items(self, items):
        # self.database.add_items(items)
        return None

    def get_location_details(self, name):
        '''Get details of a specific location'''
        pass

        return None
    def get_location_details_by_id(self, id):
        '''Get details of a specific location'''
        pass

        return None
    
    def get_item_details(self, id):
        '''Get details of a specific item'''
        pass

        return None

    def get_items_in_location(self, location_name):
        '''Get all items in a specific location'''
        pass

        return None


    def get_object_enviroment(self, id):
        '''Get the environment of a specific object'''
        pass
    
        return None
    
    def change_object_location(self, object_id, location_id):
        '''Change the location of a specific object'''
        pass

        return None
    
    def move_player_to_location(self, player_id, location_id):
        '''Move the player to a specific location'''
        pass

        return None
    
    def _add_player(self, player_name):
        '''Add a new player'''
        pass
        
        return None

    def _remove_player(self, player_id):
        '''Remove a player'''
        pass

    def move_character_to_location(self, character_id, location_id):
        '''Move a character to a specific location'''
        pass

        return None
    
    def _add_character(self, character_name):
        '''Add a new character'''
        pass

        return None
    
    def _remove_character(self, character_id):
        '''Remove a character'''
        pass
    
    def get_player_inventory(self, player_id):
        '''Get the inventory of a player'''
        pass

        return None
    
    def search_object(self, location_name, object_name):
        '''Search for an object in a location (used random result)'''
        pass

        return None
    


if __name__ != "__main__":
    MANAGER = Manager()
