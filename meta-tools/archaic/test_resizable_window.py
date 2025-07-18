#!/usr/bin/env python3
"""
Simple test to verify the resizable window functionality works
"""

import pygame
import sys
import os

def test_resizable_window():
    """Test the resizable window functionality"""
    pygame.init()
    
    # Create resizable window
    screen_width = 800
    screen_height = 600
    screen = pygame.display.set_mode((screen_width, screen_height), pygame.RESIZABLE)
    pygame.display.set_caption("Resizable Window Test")
    
    font = pygame.font.SysFont(None, 36)
    clock = pygame.time.Clock()
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            # Handle window resizing
            if event.type == pygame.VIDEORESIZE:
                screen_width, screen_height = event.w, event.h
                screen = pygame.display.set_mode((screen_width, screen_height), pygame.RESIZABLE)
                print(f"Window resized to {screen_width}x{screen_height}")
        
        # Clear screen
        screen.fill((0, 0, 0))
        
        # Draw size info
        size_text = f"Window Size: {screen_width} x {screen_height}"
        text_surface = font.render(size_text, True, (255, 255, 255))
        text_rect = text_surface.get_rect(center=(screen_width//2, screen_height//2))
        screen.blit(text_surface, text_rect)
        
        # Draw resize instructions
        inst_text = "Drag corners to resize. ESC to close."
        inst_surface = font.render(inst_text, True, (200, 200, 200))
        inst_rect = inst_surface.get_rect(center=(screen_width//2, screen_height//2 + 50))
        screen.blit(inst_surface, inst_rect)
        
        # Handle ESC key
        keys = pygame.key.get_pressed()
        if keys[pygame.K_ESCAPE]:
            running = False
        
        pygame.display.flip()
        clock.tick(60)
    
    pygame.quit()
    print("Resizable window test completed successfully!")

if __name__ == "__main__":
    test_resizable_window()
